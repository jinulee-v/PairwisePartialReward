python human_eval/generate_humaneval_dataset.py --output_file human_eval/qqp.json -c qqp_Ref -c bart_qqp_1gen -c bart_qqp_1gen_2gen+triecl -c bart_qqp_1gen_2gen+brio -c bart_qqp_1gen_2gen+mrt
python human_eval/generate_humaneval_dataset.py --output_file human_eval/mscoco.json -c mscoco_Ref -c bart_mscoco_1gen -c bart_mscoco_1gen_2gen+triecl -c bart_mscoco_1gen_2gen+brio -c bart_mscoco_1gen_2gen+mrt