"""Optional exact frozen-image features for local SmolVLA expert training.

Only deterministic image-dataset training with a frozen VLM is supported.
The exported model, observation schema and inference path stay native SmolVLA.
No object annotations or simulator state are used to construct this cache.
"""
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import types

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


class FrozenVisionDataset(Dataset):
    def __init__(self, original, features, camera_keys):
        self.original=original
        self.cached_features=features
        self.camera_keys=tuple(camera_keys)

    def __getattr__(self,name):
        if name in ('original','cached_features','camera_keys'):
            raise AttributeError(name)
        return getattr(self.original,name)

    def __len__(self):
        return len(self.original)

    @staticmethod
    def feature_key(camera):
        return 'observation.frozen_image_features.'+camera.rsplit('.',1)[-1]

    def __getitem__(self,index):
        return self.__getitems__([index])[0]

    def __getitems__(self,indices):
        items=self.original.__getitems__(list(indices))
        if self.cached_features is None:
            return items
        for item,index in zip(items,indices,strict=True):
            for camera_index,camera in enumerate(self.camera_keys):
                feature=torch.from_numpy(np.array(self.cached_features[index,camera_index],copy=True))
                # NumPy has no native bfloat16 dtype. Reinterpret the recorded
                # bits rather than numerically converting their uint16 values.
                if self.cached_features.dtype==np.uint16:
                    feature=feature.view(torch.bfloat16)
                item[self.feature_key(camera)]=feature
        return items


def _fingerprint(policy,dataset,compute):
    model=policy.model.vlm_with_expert.get_vlm_model()
    digest=hashlib.sha256()
    for branch in ('vision_model','connector'):
        module=getattr(model,branch)
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError('Frozen image cache cannot be used with trainable vision/connector parameters')
        for name,tensor in module.state_dict().items():
            digest.update((branch+'.'+name+str(tensor.dtype)+str(tuple(tensor.shape))).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    root=Path(dataset.root)
    files=[(p.relative_to(root).as_posix(),p.stat().st_size,p.stat().st_mtime_ns)
           for p in sorted((root/'data').rglob('*.parquet'))]
    details={'vision_weights_sha256':digest.hexdigest(),'dataset_root':str(root.resolve()),
        'data_files':files,'episodes':dataset.episodes,'n':len(dataset),
        'camera_keys':list(policy.config.image_features),'resize':policy.config.resize_imgs_with_padding,
        'compute':compute,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(next(policy.parameters()).device),
        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
        'prepare_images_source':inspect.getsource(type(policy).prepare_images),
        'embed_image_source':inspect.getsource(type(policy.model.vlm_with_expert).embed_image)}
    return details,hashlib.sha256(json.dumps(details,sort_keys=True,default=str).encode()).hexdigest()


def attach_frozen_vision_cache(policy,dataset,cache_root,*,compute='bf16',batch_size=32):
    """Precompute real image embeddings, then bypass image I/O during training.

    Call before Accelerate wraps the policy. No file in the source dataset is
    changed. Cache entries are committed only after a complete, hashed build.
    """
    if not policy.config.train_expert_only or policy.config.empty_cameras or policy.config.compile_model:
        raise ValueError('Cache requires expert-only, no empty cameras, and no torch.compile')
    if dataset.meta.video_keys or dataset._ensure_reader()._image_transforms is not None:
        raise ValueError('Cache supports unaugmented image datasets only')
    cameras=list(policy.config.image_features)
    if not cameras or any(dataset.meta.features[k]['dtype']!='image' for k in cameras):
        raise ValueError('Every configured camera must be a recorded image feature')
    if compute not in ('no','bf16','fp16') or batch_size<1:
        raise ValueError('Invalid cache precision or batch size')
    device=next(policy.parameters()).device
    if device.type!='cuda':
        raise ValueError('Frozen image precomputation requires the local CUDA device')
    details,key=_fingerprint(policy,dataset,compute)
    cache_root=Path(cache_root).resolve()
    cache_root.mkdir(parents=True,exist_ok=True)
    directory=cache_root/key
    if not directory.resolve().is_relative_to(cache_root):
        raise ValueError('Cache must remain inside the requested root')
    metadata_file=directory/'metadata.json'
    context=lambda: nullcontext() if compute=='no' else torch.autocast('cuda',dtype=torch.bfloat16 if compute=='bf16' else torch.float16)
    if not metadata_file.is_file():
        if directory.exists():
            raise FileExistsError('Incomplete cache preserved; inspect it before retrying')
        was_training=policy.training
        policy.eval()
        generator=torch.Generator().manual_seed(0)
        loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,generator=generator)
        with tempfile.TemporaryDirectory(prefix='vision_cache_partial_',dir=cache_root) as tmp:
            stage=Path(tmp)/'ready'
            stage.mkdir()
            array=None
            cursor=0
            try:
                with torch.random.fork_rng(devices=[device.index or 0]),torch.inference_mode(),context():
                    for batch in loader:
                        image_batch={k:batch[k].to(device) for k in cameras}
                        images,masks=policy.prepare_images(image_batch)
                        if len(images)!=len(cameras) or not all(bool(mask.all()) for mask in masks):
                            raise ValueError('Cache requires two present, unmasked recorded cameras')
                        embeddings=[policy.model.vlm_with_expert.embed_image(image) for image in images]
                        if array is None:
                            shape=(len(dataset),len(cameras),*embeddings[0].shape[1:])
                            embedding_dtype=str(embeddings[0].dtype).split('.')[-1]
                            storage_dtype='uint16' if embedding_dtype=='bfloat16' else 'float32'
                            array=np.lib.format.open_memmap(stage/'features.npy',mode='w+',dtype=np.dtype(storage_dtype),shape=shape)
                        count=embeddings[0].shape[0]
                        for i,embedding in enumerate(embeddings):
                            if str(embedding.dtype).split('.')[-1]!=embedding_dtype:
                                raise ValueError('Camera embedding dtypes differ')
                            encoded=embedding.contiguous().view(torch.uint16) if storage_dtype=='uint16' else embedding.float()
                            array[cursor:cursor+count,i]=encoded.cpu().numpy()
                        cursor+=count
                        if cursor%max(batch_size*20,1)==0 or cursor==len(dataset):
                            print(f'Frozen RGB embeddings: {cursor}/{len(dataset)}',flush=True)
                if array is None or cursor!=len(dataset):
                    raise RuntimeError('Incomplete frozen image feature build')
                shape=tuple(array.shape)
                array.flush()
                array._mmap.close()
                array=None
                with (stage/'features.npy').open('rb') as stream:
                    sha=hashlib.file_digest(stream,'sha256').hexdigest()
                metadata={'key':key,'source':details,'shape':shape,'storage_dtype':storage_dtype,
                    'storage_format':'bfloat16_bits_uint16' if storage_dtype=='uint16' else 'float32',
                    'embedding_dtype':embedding_dtype,'features_sha256':sha,
                    'scope':'Real deterministic frozen RGB encoder features; no privileged annotations; inference remains RGB2/state16/action7.'}
                (stage/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')
                stage.rename(directory)
            finally:
                if array is not None:
                    array._mmap.close()
                policy.train(was_training)
    metadata=json.loads(metadata_file.read_text(encoding='utf-8'))
    if metadata['key']!=key or metadata['shape'][0]!=len(dataset):
        raise ValueError('Frozen image cache does not match this dataset/encoder')
    with (directory/'features.npy').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=metadata['features_sha256']:
            raise ValueError('Frozen image feature cache hash mismatch')
    array=np.load(directory/'features.npy',mmap_mode='r')
    storage_dtype=metadata['storage_dtype']
    if storage_dtype not in ('float32','uint16'):
        raise ValueError('Unsupported frozen image cache storage dtype')
    if storage_dtype=='uint16' and (metadata['embedding_dtype']!='bfloat16' or
                                    metadata.get('storage_format')!='bfloat16_bits_uint16'):
        raise ValueError('uint16 cache entries must explicitly encode bfloat16 bits')
    if list(array.shape)!=metadata['shape'] or array.dtype!=np.dtype(storage_dtype):
        raise ValueError('Frozen image cache shape/dtype mismatch')
    # Keep metadata and episode windows native; project only the in-memory Arrow
    # view so the loader reads robot state/actions/text without decoding PNGs.
    reader=dataset._ensure_reader()
    reader.hf_dataset=dataset.hf_dataset.remove_columns(cameras)
    if reader.delta_indices is not None:
        reader.delta_indices={k:v for k,v in reader.delta_indices.items() if k not in cameras}
    proxy=FrozenVisionDataset(dataset,array,cameras)
    original_prepare=policy.prepare_images
    original_embed=policy.model.vlm_with_expert.embed_image
    dtype=getattr(torch,metadata['embedding_dtype'])
    feature_shape=tuple(metadata['shape'][2:])

    def prepare_images(self,batch):
        keys=[FrozenVisionDataset.feature_key(k) for k in cameras]
        if any(k in batch for k in keys):
            if not all(k in batch for k in keys):
                raise ValueError('Partial frozen camera features are not valid inputs')
            features=[batch[k].to(dtype=dtype) for k in keys]
            if any(tuple(x.shape[1:])!=feature_shape for x in features):
                raise ValueError('Invalid frozen image embedding shape')
            return features,[torch.ones(x.shape[0],dtype=torch.bool,device=x.device) for x in features]
        return original_prepare(batch)

    def embed_image(self,image):
        if image.ndim==3 and tuple(image.shape[1:])==feature_shape:
            return image
        return original_embed(image)

    policy.prepare_images=types.MethodType(prepare_images,policy)
    policy.model.vlm_with_expert.embed_image=types.MethodType(embed_image,policy.model.vlm_with_expert)
    return proxy,metadata


def install_frozen_vision_training(lerobot_train,pipeline_config,cache_root,*,compute='bf16'):
    """Scope the dataset/model integration to one native LeRobot training process."""
    if pipeline_config.num_workers!=0 or pipeline_config.rename_map:
        raise ValueError('Frozen vision training requires num_workers=0 and native camera names')
    if pipeline_config.dataset.image_transforms.enable:
        raise ValueError('Frozen vision training requires disabled image augmentation')
    from lerobot.configs import FeatureType,NormalizationMode
    if pipeline_config.policy.normalization_mapping[FeatureType.VISUAL]!=NormalizationMode.IDENTITY:
        raise ValueError('Frozen RGB cache requires identity visual normalization')
    original_data=lerobot_train.make_train_eval_datasets
    original_policy=lerobot_train.make_policy
    holder={}

    def make_datasets(*args,**kwargs):
        dataset,evaluation=original_data(*args,**kwargs)
        proxy=FrozenVisionDataset(dataset,None,())
        holder['dataset']=proxy
        return proxy,evaluation

    def make_policy(*args,**kwargs):
        policy=original_policy(*args,**kwargs)
        proxy=holder['dataset']
        cached,metadata=attach_frozen_vision_cache(policy,proxy.original,cache_root,
            compute=compute,batch_size=pipeline_config.batch_size)
        proxy.cached_features=cached.cached_features
        proxy.camera_keys=cached.camera_keys
        output=Path(pipeline_config.output_dir)
        output.mkdir(parents=True,exist_ok=True)
        (output/'frozen_vision_cache.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        return policy

    lerobot_train.make_train_eval_datasets=make_datasets
    lerobot_train.make_policy=make_policy
