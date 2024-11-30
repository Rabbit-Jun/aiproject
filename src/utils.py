import os
import shutil
import random
from pathlib import Path

`def create_test_dataset(source_dir: str, target_dir: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Train 데이터셋에서 일부를 분리하여 테스트 데이터셋 생성
    
    Args:
        source_dir: 원본 train 데이터 경로
        target_dir: 새로 만들 test 데이터 경로
        test_ratio: 테스트 데이터 비율 (기본값: 0.2)
        seed: 랜덤 시드
    """
    random.seed(seed)
    
    # 대상 디렉토리 생성
    os.makedirs(target_dir, exist_ok=True)
    
    # 각 클래스별로 처리
    for class_name in os.listdir(source_dir):
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        
        # 클래스가 디렉토리인지 확인
        if not os.path.isdir(source_class_dir):
            continue
            
        # 테스트 데이터 저장할 디렉토리 생성
        os.makedirs(target_class_dir, exist_ok=True)
        
        # 해당 클래스의 모든 이미지 파일 목록
        all_files = [f for f in os.listdir(source_class_dir) if f.endswith('.jpg')]
        
        # 테스트 데이터로 사용할 파일 수 계산
        num_test = int(len(all_files) * test_ratio)
        
        # 테스트 데이터로 사용할 파일 무작위 선택
        test_files = random.sample(all_files, num_test)
        
        # 파일 이동
        for filename in test_files:
            source_path = os.path.join(source_class_dir, filename)
            target_path = os.path.join(target_class_dir, filename)
            shutil.move(source_path, target_path)
            
        print(f"{class_name}: {num_test}개 파일을 테스트 데이터로 이동했습니다.")

if __name__ == "__main__":
    # 경로 설정
    train_dir = "./dataset/flowers/train"
    test_dir = "./dataset/flowers/test"
    
    # 기존 test 폴더 백업
    if os.path.exists("./dataset/flowers/test"):
        backup_dir = "./dataset/flowers/test_backup"
        print(f"기존 test 폴더를 {backup_dir}로 백업합니다.")
        shutil.move("./dataset/flowers/test", backup_dir)
    
    # 새로운 테스트 데이터셋 생성
    create_test_dataset(train_dir, test_dir, test_ratio=0.2)
    `
    # 결과 출력
    print("\n데이터셋 분할 완료!")
    print("\n각 클래스별 파일 수:")
    for class_name in os.listdir(train_dir):
        train_count = len(os.listdir(os.path.join(train_dir, class_name)))
        test_count = len(os.listdir(os.path.join(test_dir, class_name)))
        print(f"{class_name}: train={train_count}, test={test_count}")