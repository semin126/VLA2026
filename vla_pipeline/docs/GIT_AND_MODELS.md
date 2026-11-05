# 코드와 모델을 나누어 전달하기

코드는 이 GitHub 저장소에서 공유합니다. 설치는 [SETUP_STUDENT.md](../SETUP_STUDENT.md), 연구 안내는 [STUDENT_RESEARCH_GUIDE.md](STUDENT_RESEARCH_GUIDE.md)를 따릅니다. 조교용 참조 구현은 저장소 밖에 있으며 학생 배포 대상이 아닙니다.

## 코드 받기와 돌려주기

처음에는 `git clone https://github.com/semin126/VLA2026.git`으로 받습니다. 이미 받았다면 저장소에서 다음을 실행합니다.

```powershell
git status
git pull --ff-only
```

수정 중인 파일이 있으면 먼저 본인 작업을 별도 브랜치에 커밋합니다. 충돌이나 fast-forward 불가가 나오면 작업을 버리는 reset 명령을 쓰지 말고 조교와 변경 내역을 비교합니다. 학생별 브랜치를 쓰면 서로의 미완성 코드를 덮어쓰지 않고 검토할 수 있습니다.

```powershell
git switch -c student/MY_NAME
git add vlapipe/student_design.py
git commit -m "Implement and explain my planning policy"
git push -u origin student/MY_NAME
```

실험 설정과 분석 코드도 함께 커밋하되, 개인 정보·토큰·가상환경·대형 데이터·weight를 추가하지 않습니다. 실험 기록에는 `git rev-parse HEAD` 결과와 커밋 이후 수정 여부를 남깁니다. `git`이 인식되지 않으면 Git 설치 후 VS Code와 터미널을 다시 엽니다.

## weight는 별도 배포

일반 GitHub 저장소는 100MiB 초과 파일을 차단합니다. 현재 VLA 연결 시험의 weight만 약 865MiB이므로 일반 `git add`·push 대상으로 삼지 않습니다. [GitHub 공식 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

공유 Drive의 버전별 폴더 또는 모델 저장소를 사용하고 Git에는 배포 명세만 둡니다. Git LFS도 가능하지만 팀의 저장 용량·다운로드 할당량을 먼저 확인합니다. 현재 `outputs/`와 `data/`는 Git에서 제외됩니다.

weight 파일 하나만 보내지 말고 **실제로 재로딩에 사용한 pretrained_model 폴더 전체**와 필요한 기반 모델의 정확한 ID·revision을 전달합니다. 구성에 따라 config, 이미지 전처리, tokenizer, 상태·액션 정규화 통계, policy 전후처리 설정이 필요합니다. 학습 재개용 optimizer 상태는 추론용 배포와 구분합니다.

배포 명세에 아래를 기록합니다.

- 모델 ID·기반 revision, 코드 commit과 수정 여부, 라이브러리 버전.
- 학습 데이터 manifest와 분할, 학습 스텝·설정, 선택 기준.
- 파일 목록·크기·SHA256, 저장 폴더, 다운로드 주소와 접근 방법.
- 입력 카메라·상태 차원·액션 의미·정규화 및 로딩 명령.
- 실제 평가 결과와 한계. `connection-smoke`와 `task-validated`를 구분.

PowerShell에서 파일 해시는 `Get-FileHash -Algorithm SHA256 경로`로 계산할 수 있습니다. 받은 사람도 해시를 비교한 뒤 저장 폴더로 실행합니다. 새 버전은 새 폴더로 배포하고 이전 실험의 모델을 조용히 교체하지 않습니다.

현재 1~2스텝 weight는 **연결 시험 산출물**이며 검증된 택배 분류 모델이 아닙니다. 충분한 학습과 독립 장면 평가를 거친 뒤 배포 명세에 결과를 추가합니다. 공유 Drive 주소는 실제 업로드·권한 확인 후 기록합니다.
