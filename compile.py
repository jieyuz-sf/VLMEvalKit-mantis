import argparse
import os
import json

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resdir', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    print('\n')
    print(f"{args.resdir}")

    benchmark_list = [
        'SEEDBench_IMG_acc.csv',
        'SEEDBench2_acc.csv',
        'MMBench_DEV_EN_acc.csv',
        'MMStar_acc.csv', 
        'MME_score.csv',
        "cvbench_acc.csv",
        'MMVet_gpt-4-turbo_score.csv',
        'LLaVABench_score.csv',
        'MMMU_DEV_VAL_acc.csv',
        'MathVista_MINI_gpt-4-turbo_score.csv',
        'AI2D_TEST_acc.csv',
        'ScienceQA_TEST_acc.csv',
        'POPE_score.csv',
        'HallusionBench_score.csv',
        'TextVQA_VAL_acc.csv',
        'DocVQA_VAL_acc.csv',
        'ChartQA_TEST_acc.csv',
        'OCRBench_score.json',
        'RealWorldQA_acc.csv'
    ]
    benchmark_names = [
    "SEED",
    "SEEDv2",
    "MMB",
    "MMStar",
    "MME",
    "CVB-2D",
    "CVB-3D",
    "MMVet",
    "LlavaW",
    "MMMU",
    "Math",
    "AI2D",
    "SciQA",
    "POPE",
    "HalBch",
    "TVQA",
    "DocVQA",
    "ChartQA",
    "OCRBch",
    "RealWorldQA"
    ]
    res_list = []
    for benchmark in benchmark_list:
        fn = os.path.join(args.resdir, f"{benchmark}")
        if not os.path.exists(fn):
            fn = os.path.join(args.resdir, f"{args.resdir}_{benchmark}")
        acc = 0
        if os.path.exists(fn):
            if 'OCRBench_score' in fn:
                ocr_res_dict = json.load(open(fn, 'r'))
                acc = ocr_res_dict['Final Score Norm']
                res_list.append(f"{acc: .1f}")
                continue
            df = pd.read_csv(fn)
            if 'POPE' in benchmark:
                acc = df[df['split'] == 'Overall']['Overall'][0]
            elif 'HallusionBench' in benchmark:
                acc = df[df['split'] == 'Overall']['aAcc'][0]
            elif 'MMMU' in benchmark:
                acc = df[df['split'] == 'validation']['Overall'].values[0]
                acc *= 100
            elif 'MMVet' in benchmark:
                acc = df[df['Category'] == 'Overall']['acc'].values[0]
            elif 'MathVista' in benchmark:
                acc = df[df['Task&Skill'] == 'Overall']['acc'].values[0]
            elif 'LLaVABench' in benchmark:
                acc = df[df['split'] == 'overall']['Relative Score (main)'].values[0]
            elif 'MME' in benchmark:
                # acc = [
                #         df['perception'][0] + df['reasoning'][0],
                #         df['perception'][0], 
                #         df['reasoning'][0]
                #     ]
                acc = 100 * (df['perception'][0] + df['reasoning'][0]) / 2800
            elif 'cvbench' in benchmark:
                acc = [
                    100 * (df["2D_Count"][0] * 788 + df["2D_Relation"][0] * 650) / (788 + 650),
                    100 * ((df["3D_Depth"][0] + df["3D_Distance"][0]) / 2)
                ]
            else:
                try:
                    acc = df["Overall"][0]
                    if acc < 1.0:
                        acc *= 100
                except Exception as e:
                    print(f"Error parsing result for {benchmark}")
                    print(e)
            if isinstance(acc, list):
                acc = [f"{a: .0f}" for a in acc]
                res_list.extend(acc)
            else:
                res_list.append(f"{acc: .1f}")
        else:
            print(f"File non-exist: {fn}")
            res_list.append(f"{acc: .1f}")
    
    print('	'.join(benchmark_names))
    print('	'.join(res_list))
