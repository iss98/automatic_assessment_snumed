import argparse
import torch
from data import item_list
from model import model_list
from tok import tokenizer_list

"""
하이퍼파라미터 튜잉을 위한 방법
"""

def get_cfg():
    parser = argparse.ArgumentParser(description="")
    """
    모델 설정
    """
    parser.add_argument(
        "--item", type = str, choices = item_list, help = "item number"
    ) #사용할 문항 지정 예) 1-1, 3-2, 모두 사용하는 경우 all 입력
    parser.add_argument(
        "--model", type = str, default = "rnn", choices = model_list, help = "model type"
    ) #사용할 모델 지정 rnn, lstm, att (att는 multihead attention 기반 모델)
    parser.add_argument(
        "--tokenizer", type = str, default = "sp", choices = tokenizer_list, help="choose tokenizer"
    ) #tokenizer 선택하기 선택지는 2개 sp(SentencePieceTokenizer) , bw(BertWordPieceTokenizer)
    parser.add_argument(
        "--prjname", type = str, default = "Automatic Assessment", help = "name of the project for wandb"
    ) #wandb 홈페이지에서 사용할 프로젝트 이름 설정할 필요 없음
    """
    모델의 구조 설정하는 하이퍼파라미터
    """
    parser.add_argument(
        "--vs", type = int, default = 60, help = "vocab size"
    ) #적절한 vocab size 설정하기, 문제에 사용되는 토큰이 많지 않아 작게 설정하는 것을 추천
    parser.add_argument(
        "--emb", type = int, default = 16, help = "dimension of embedding"
    ) #토큰들을 벡터로 만들어 줄때, 벡터의 차원 설정
    parser.add_argument(
        "--hidden", type = int, default = 32, help = "dimension of main layer"
    ) #모델의 메인 layer의 차원 설정
    """
    학생 응답의 최대 길이 설정
    """
    parser.add_argument(
        "--ml", type = int, default = 100, help = "max len"
    ) 
    """
    bwotkenizer train 시 사용되는 하이퍼파라미터, sp사용하는 경우는 설정할 필요없음
    """
    parser.add_argument(
        "--la", type = int, default = 6000, help = "limit alphabet"
    ) 
    parser.add_argument(
        "--mf", type = int, default = 1, help = "min frequencey"
    )
    parser.add_argument(
        "--lc", type = bool, default = False, help = "lowercase"
    )
    parser.add_argument(
        "--sa", type = bool, default = False, help = "strip accents"
    )
    """
    학습에 필요한 하이퍼파라미터
    """
    parser.add_argument(
        "--device", type = str, default = "cuda:0" if torch.cuda.is_available() else "cpu", help = "GPU"
    ) #그래픽카드 지정, 구글 코랩 환경인 경우 추가적인 조작 필요 없음, 맥 사용시 mps로 설정
    parser.add_argument(
        "--testsplit", type = float, default = 0.2, help = "split testset"
    )
    # parser.add_argument(
    #     "--valsplit", type = float, default = 0.25, help = "split validset"
    # )
    parser.add_argument(
        "--lr", type = float, default = 1e-4, help = "learning rate"
    )
    parser.add_argument(
        "--mt", type = float, default = 0.9, help = "momentum for SGD"
    )
    parser.add_argument(
        "--epoch", type = str, default = 100, help = "number of epochs"
    )
    parser.add_argument(
        "--testevery", type = str, default = 10, help = "test period"
    )
    parser.add_argument(
        "--bs", type = str, default = 16, help = "batch size"
    )
    return parser.parse_args()
