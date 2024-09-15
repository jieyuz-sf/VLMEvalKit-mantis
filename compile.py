import argparse
import os
import json

import pandas as pd


def parse_args():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--resdir', type=str, required=True)
	parser.add_argument('--resdir', type=str, default='/export/share/jieyu/mantis_ckpt/Mantis-8B-siglip-llama3-pretraind')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	print('\n')
	print(f"{args.resdir}")

	benchmark_list = [
		'BLINK_acc.csv',
		'MMT-Bench_VAL_MI_acc.csv',
		'SEEDBench_IMG_acc.csv',
		'SEEDBench2_acc.csv',
		'MMBench_DEV_EN_acc.csv',
		'MMStar_acc.csv',
		'MME_score.csv',
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
		'RealWorldQA_acc.csv',
		'QBench2_acc.csv',
		"cvbench_acc.csv",
	]
	benchmark_names = [
		"BLINK",
		"MMT",
		"SEED",
		"SEEDv2",
		"MMB",
		"MMStar",
		"MME",
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
		"RealWorldQA",
		"QBench2",
		"CVB-2D",
		"CVB-3D",
	]

	to_print = []
	to_print.append('\t'.join(['model'] + benchmark_names))
	res_json = {'names': benchmark_names}

	for d in os.listdir(args.resdir):

		if os.path.isdir(os.path.join(args.resdir, d, 'checkpoint-final')):

			resdir = os.path.join(args.resdir, d, 'checkpoint-final')
			res_list = []
			for benchmark in benchmark_list:
				fn = os.path.join(resdir, f"{benchmark}")
				if not os.path.exists(fn):
					fn = os.path.join(resdir, f"{resdir}_{benchmark}")
				acc = 0
				if os.path.exists(fn):
					if 'OCRBench_score' in fn:
						ocr_res_dict = json.load(open(fn, 'r'))
						acc = ocr_res_dict['Final Score Norm']
						res_list.append(f"{acc: .2f}")
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
							print(f"Error parsing result for {benchmark}", resdir)
							print(e)
					if isinstance(acc, list):
						acc = [f"{a: .0f}" for a in acc]
						res_list.extend(acc)
					else:
						res_list.append(f"{acc: .2f}")
				else:
					print(f"File non-exist: {fn}")
					res_list.append(f"{acc: .2f}")

			to_print.append('\t'.join([d] + res_list))
			res_json[d] = res_list

	print('\n'.join(to_print))
	with open("m_res.csv", 'w') as f:
		f.write('\n'.join(to_print))

	from huggingface_hub import HfApi
	from huggingface_hub import login
	from datetime import datetime

	login('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')

	api = HfApi()
	api.upload_file(
		path_or_fileobj="m_res.csv",
		path_in_repo=f"m_res-{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.csv",
		repo_id="jieyuz2/res",
		repo_type="dataset",
	)
