source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/TVB/envs/body4d
cd /root/TVB/SAM4Dcap/pipeline2
python run_prepare_trc.py \
  --dynamic-video /root/TVB/SAM4Dcap/opencap/data/subject2/Session0/Videos/Cam1/InputMedia/DJ1/DJ1_syncdWithMocap.mp4 \
  --static-video /root/TVB/SAM4Dcap/opencap/data/subject2/Session0/Videos/Cam2/InputMedia/static1/static1_syncdWithMocap.mp4

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/TVB/envs/opensim

python /root/TVB/SAM4Dcap/pipeline2/run_scale.py

python /root/TVB/SAM4Dcap/pipeline2/run_ik.py


python /root/TVB/SAM4Dcap/pipeline2/run_vis.py

cd /root/TVB/SAM4Dcap/pipeline2
python -m http.server 8093 --bind 127.0.0.1
