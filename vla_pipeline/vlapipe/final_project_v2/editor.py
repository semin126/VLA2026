"""Optional Gradio coordinate editor; notebook/CLI PNG previews always work."""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

from .common import dump
from .scenes import freeze, preview, read_scene, template, validate_layout


def launch(scene_path, work, share=False):
    import gradio as gr
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    spec = read_scene(scene_path)

    def rows(s):
        return [[n, *p] for n, p in s['layout'].items()]

    def edit(table, category, state):
        value = copy.deepcopy(state)
        value.pop('initial_state', None)
        value.pop('state_sha256', None)
        value['category'] = category
        value['layout'] = {str(r[0]): [float(x) for x in r[1:4]] for r in table}
        return value

    def inspect(table, category, state):
        value = edit(table, category, state)
        report = validate_layout(value)
        if report['errors']:
            return None, json.dumps(report, ensure_ascii=False, indent=2)
        folder = work / ('preview_' + uuid.uuid4().hex[:8])
        im = preview(value, folder)
        return str(im), json.dumps(report, ensure_ascii=False, indent=2)

    def save(table, category, state):
        value = edit(table, category, state)
        # Any edited source becomes development data: never silently edit a frozen test.
        value['split'] = 'development'
        value['id'] = 'student_' + uuid.uuid4().hex[:8]
        # Keep family lineage when editing an existing scene to prevent split leakage.
        value = freeze(value)
        dest = work / (value['id'] + '.json')
        dump(dest, value)
        preview(value, dest.with_suffix(''))
        return str(dest), json.dumps(value['settled_diagnostics'], ensure_ascii=False, indent=2)

    def fresh(category):
        value = template(category, seed=int(uuid.uuid4().hex[:6], 16))
        return rows(value), value

    with gr.Blocks(title='Final project ver2.0 — Scene designer') as app:
        gr.Markdown('# Final project ver2.0 — Scene designer\n좌표는 m, 회전은 rad입니다. 배치 변경 → 미리보기 → 검증·저장 순서로 사용하세요. 물체 ID는 유지하세요.')
        state = gr.State(spec)
        category = gr.Dropdown(['open', 'blocked', 'mixed'], value=spec['category'], label='설계 의도')
        fresh_button = gr.Button('새 배치 예시')
        table = gr.Dataframe(headers=['object', 'x', 'y', 'yaw'], datatype=['str', 'number', 'number', 'number'],
                             value=rows(spec), type='array', interactive=True, label='물체 배치')
        view = gr.Button('실제 MuJoCo 렌더·검사')
        image = gr.Image(label='상단·정면·측면·손목', type='filepath')
        diagnostics = gr.Textbox(lines=10, label='진단 — 실제 파지 가능성은 실행해서 확인')
        save_button = gr.Button('초기 상태 고정·새 개발 씬으로 저장')
        saved = gr.File(label='저장한 씬 JSON — Colab에 업로드 가능')
        fresh_button.click(fresh, [category], [table, state])
        view.click(inspect, [table, category, state], [image, diagnostics], concurrency_limit=1)
        save_button.click(save, [table, category, state], [saved, diagnostics], concurrency_limit=1)
    # Serialize MuJoCo/global numbered configuration changes across callbacks.
    return app.queue(default_concurrency_limit=1).launch(share=share, inbrowser=False, allowed_paths=[str(work)])
