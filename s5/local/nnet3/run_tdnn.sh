#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
dir=exp/nnet3/tdnn
trans_dir=exp/tri3_ali
egs_dir=exp/nnet3/egs
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
if [ $stage -le 0 ]; then
mfccdir=mfcc
for x in train dev test; do
  ./utils/copy_data_dir.sh data/$x data/${x}_hires
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 --mfcc-config conf/mfcc_hires.conf \
    data/${x}_hires exp/make_mfcc/${x}_hires $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/${x}_hires exp/make_mfcc/${x} $mfccdir || exit 1;
done
exit 0;
fi
if [ $stage -le 1 ]; then
./steps/nnet3/get_egs.sh --cmd "$train_cmd" --config conf/get_egs.conf --nj 4 \
  data/train_hires exp/tri3_ali exp/nnet3/tdnn/egs || exit 1;

./steps/nnet3/tdnn/make_configs.py \
   --feat-dir data/train_hires \
   --ali-dir exp/tri3_ali \
   --splice-indexes "-2,2 -1,2 -3,3 -7,2 0" \
   --relu-dim 1024 exp/nnet3/tdnn/configs || exit 1;
fi

if [ $stage -le 8 ]; then
  steps/nnet3/train_dnn.py \
                    --feat.cmvn-opts "--norm-means=true --norm-vars=true" \
                    --egs.frames-per-eg 3 \
                    --egs.transform_dir $trans_dir \
                    --egs.dir exp/nnet3/tdnn/egs \
                    --trainer.shuffle-buffer-size 50000 \
                    --trainer.num-epochs 15 \
                    --trainer.add-layers-period 1 \
                    --trainer.optimization.minibatch-size 512 \
                    --trainer.optimization.initial-effective-lrate 0.0017 \
                    --trainer.optimization.final-effective-lrate 0.00017 \
                    --trainer.optimization.num-jobs-initial 1 \
                    --trainer.optimization.num-jobs-final 1 \
                    --stage $train_stage \
                    --use-gpu true \
                    --cleanup true \
                    --feat-dir data/train_hires \
                    --lang data/lang \
                    --ali-dir exp/tri3_ali \
                    --dir $dir || exit 1;
fi


if [ $stage -le 9 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding.
  graph_dir=exp/tri3/graph
  # use already-built graphs.
  for d in dev test; do
    steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" --stage 3 \
        $graph_dir data/${d}_hires $dir/decode_${d} || exit 1;
  done
fi


exit 0;

# results:
grep WER exp/nnet3/nnet_tdnn_a/decode_{tgpr,bd_tgpr}_{eval92,dev93}/scoring_kaldi/best_wer
exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/scoring_kaldi/best_wer:%WER 6.03 [ 340 / 5643, 74 ins, 20 del, 246 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_eval92/wer_13_1.0
exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/scoring_kaldi/best_wer:%WER 9.35 [ 770 / 8234, 162 ins, 84 del, 524 sub ] exp/nnet3/nnet_tdnn_a/decode_tgpr_dev93/wer_11_0.5
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/scoring_kaldi/best_wer:%WER 3.81 [ 215 / 5643, 30 ins, 18 del, 167 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_eval92/wer_10_1.0
exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/scoring_kaldi/best_wer:%WER 6.74 [ 555 / 8234, 69 ins, 72 del, 414 sub ] exp/nnet3/nnet_tdnn_a/decode_bd_tgpr_dev93/wer_11_0.0
b03:s5:
