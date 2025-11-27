#!/bin/bash

# GPU 모니터링 및 자동 학습 시작 스크립트
# GPU 2,3번이 VRAM 10% 이하로 내려가고 1분간 증가 추세가 없으면 학습 시작

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== GPU 모니터링 및 자동 학습 스크립트 시작 ===${NC}"
echo "GPU 2,3번의 VRAM이 10% 이하로 내려가는지 모니터링합니다..."
echo "조건 충족 시 자동으로 학습을 시작합니다."
echo ""

# 작업 디렉토리 설정
WORK_DIR="/home/byounggun/RadarDistill/tools"
cd "$WORK_DIR" || exit 1

# 환경 활성화
echo -e "${YELLOW}가상환경 활성화 중...${NC}"
source ~/radardistill_env/bin/activate

# nvidia-smi가 설치되어 있는지 확인
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi를 찾을 수 없습니다.${NC}"
    exit 1
fi

# GPU VRAM 사용률을 체크하는 함수 (%)
get_gpu_memory_usage() {
    local gpu_id=$1
    # VRAM 사용률 퍼센트 가져오기
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$gpu_id" | awk '{printf "%.1f\n", ($1/$2)*100}'
}

# 두 GPU 모두 30% 이하인지 확인하는 함수
check_both_gpus_free() {
    local gpu2_usage=$(get_gpu_memory_usage 2)
    local gpu3_usage=$(get_gpu_memory_usage 3)
    
    echo -e "GPU 2: ${gpu2_usage}% | GPU 3: ${gpu3_usage}%"
    
    # bc를 사용하여 부동소수점 비교
    if (( $(echo "$gpu2_usage < 10.0" | bc -l) )) && (( $(echo "$gpu3_usage < 10.0" | bc -l) )); then
        return 0  # 둘 다 10% 미만
    else
        return 1  # 하나라도 10% 이상
    fi
}

# 메인 모니터링 루프
echo -e "${YELLOW}GPU 모니터링 시작...${NC}"
echo ""

while true; do
    current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    if check_both_gpus_free; then
        echo -e "${GREEN}[$current_time] GPU 2,3번 모두 10% 이하 감지!${NC}"
        echo -e "${YELLOW}1분간 VRAM 증가 추세를 확인합니다...${NC}"
        
        # 초기 VRAM 값 저장
        initial_gpu2=$(get_gpu_memory_usage 2)
        initial_gpu3=$(get_gpu_memory_usage 3)
        
        # 1분(60초) 동안 5초마다 체크
        stable=true
        for i in {1..12}; do
            sleep 5
            current_gpu2=$(get_gpu_memory_usage 2)
            current_gpu3=$(get_gpu_memory_usage 3)
            
            echo "  체크 $i/12: GPU 2: ${current_gpu2}% | GPU 3: ${current_gpu3}%"
            
            # 10% 초과하거나 초기값보다 5% 이상 증가하면 안정적이지 않음
            if (( $(echo "$current_gpu2 > 10.0" | bc -l) )) || (( $(echo "$current_gpu3 > 10.0" | bc -l) )); then
                echo -e "${RED}  GPU 사용률이 10%를 초과했습니다. 모니터링을 재시작합니다.${NC}"
                stable=false
                break
            fi
            
            if (( $(echo "$current_gpu2 > $initial_gpu2 + 5.0" | bc -l) )) || (( $(echo "$current_gpu3 > $initial_gpu3 + 5.0" | bc -l) )); then
                echo -e "${RED}  VRAM이 증가하는 추세입니다. 모니터링을 재시작합니다.${NC}"
                stable=false
                break
            fi
        done
        
        if [ "$stable" = true ]; then
            echo -e "${GREEN}=== 조건 충족! 학습을 시작합니다. ===${NC}"
            echo ""
            
            # output 폴더 이름 변경
            # if [ -d "/home/byounggun/RadarDistill/output" ]; then
            #     echo -e "${YELLOW}기존 output 폴더를 output2로 변경합니다...${NC}"
            #     mv /home/byounggun/RadarDistill/output /home/byounggun/RadarDistill/output2
            #     echo -e "${GREEN}폴더 이름 변경 완료!${NC}"
            # else
            #     echo -e "${YELLOW}output 폴더가 존재하지 않습니다. 건너뜁니다.${NC}"
            # fi
            
            echo ""
            echo -e "${GREEN}학습 시작...${NC}"
            echo "명령어: CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_train.sh 2 --cfg_file cfgs/radar_distill/radar_distill_train.yaml --pretrained_model ../ckpt/pillarnet_fullset_init.pth --fix_random_seed --extra_tag ddmup"
            echo ""
            
            # 학습 실행
            CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_train.sh 2 --cfg_file cfgs/radar_distill/radar_distill_train.yaml --pretrained_model ../ckpt/pillarnet_fullset_init.pth --fix_random_seed --extra_tag ddmup
            
            echo -e "${GREEN}=== 스크립트 종료 ===${NC}"
            exit 0
        fi
    else
        echo -e "[$current_time] GPU 사용 중 - 대기..."
    fi
    
    # 10초 대기 후 다시 체크
    sleep 10
done
