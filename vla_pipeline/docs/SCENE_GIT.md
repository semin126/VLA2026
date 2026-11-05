# 학생 씬을 Git으로 계속 공유하기

`outputs/` 전체를 Git에 올리지 않고 다음 폴더만 공유 대상으로 사용합니다. 하위 폴더의 씬 JSON, 고정 상태, 미리보기 등도 포함합니다.

| 폴더 | 용도 |
|---|---|
| `outputs/valid_specs/` | 기존 학생 검증 씬 원본 JSON |
| `outputs/valid_suite/` | 기존 고정 검증 씬·suite.json·미리보기 |
| `outputs/shared_scenes/` | 앞으로 공유할 새 씬 묶음 |

학습 모델, 실행 로그, 일반 trial 결과는 기존 위치를 유지합니다. 위 공유 폴더에 모델이나 일반 실험 결과를 섞지 않습니다. 생성한 씬이 다른 출력 폴더에 있으면 공유 폴더로 등록한 뒤 커밋합니다.

## 새 묶음을 만드는 예

`vla_pipeline` 폴더에서 다음 명령을 실행합니다. 출력 경로는 기존 파일을 덮어쓰지 않도록 새 버전 이름을 사용합니다.

```powershell
$py = Join-Path $env:USERPROFILE 'miniforge3\envs\vla2026\python.exe'
& $py scripts/final_project_v2.py register-scenes --scene outputs/valid_suite/validation/scene_0000.json --scene outputs/valid_suite/validation/scene_0001.json --scene outputs/valid_suite/validation/scene_0002.json --out outputs/shared_scenes/team01_validation_v1
```

본인이 만든 다른 frozen 씬의 JSON 경로로 `--scene` 값을 바꿀 수 있습니다. register-scenes는 suite.json과 미리보기를 만들고 분할·고정 상태를 확인합니다. 새로운 묶음을 만든다고 validation 씬을 학습용이나 최종 시험용으로 바꾸지는 않습니다.

## 올리고 받기

아래 명령도 `vla_pipeline` 폴더 기준입니다. 첫 배포에는 `.gitignore` 변경도 함께 커밋해야 다른 PC에 같은 규칙이 적용됩니다.

```powershell
git status --short -- .gitignore outputs/valid_specs outputs/valid_suite outputs/shared_scenes
git add -- .gitignore outputs/valid_specs outputs/valid_suite
# 새 묶음은 실제 생성한 경로를 추가합니다.
# git add -- outputs/shared_scenes/team01_validation_v1
git diff --cached --stat
git commit -m "Share validation scene assets"
git push origin HEAD
```

공유 폴더 안의 파일도 git add/commit을 해야 업로드됩니다. 폴더가 아직 없다면 그 경로는 git add에서 빼세요. `git add .`로 다른 수정과 모델을 함께 추가하지 않고 씬 경로만 선택합니다.

받는 쪽은 같은 브랜치에서 수정 사항을 먼저 별도 커밋한 후 `git pull --ff-only`를 실행합니다. 현재처럼 조교 수정과 학생 커밋이 함께 있는 상태에서는 먼저 브랜치를 정리해야 하며, 위 명령이 자동으로 병합·충돌을 해결하지는 않습니다.

노트북에서 받은 묶음을 사용하려면 설정 셀에 다음처럼 지정합니다.

```python
SCENES_ROOT = 'outputs/valid_suite'
# 또는 'outputs/shared_scenes/team01_validation_v1'
```

학생 브랜치의 최신 씬을 파일 단위로 받아 둔 것과 로컬 브랜치를 해당 커밋까지 병합한 것은 서로 다른 상태입니다. 브랜치와 미커밋 변경은 `git status`로 확인하세요.
