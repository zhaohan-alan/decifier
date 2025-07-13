python bin/evaluate.py \
  --model-ckpt deCIFer_model/ckpt.pt \
  --dataset-path data/noma/noma-10k/serialized/test.h5 \
  --dataset-name noma-10k \
  --out-folder eval_files \
  --add-composition \
  --add-spacegroup