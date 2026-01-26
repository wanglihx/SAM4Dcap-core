cd /root/TVB
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/TVB/envs/opencap

python /root/TVB/SAM4Dcap/opencap_clean/code/run_subject2_session0_dj1.py --phase all

python /root/TVB/SAM4Dcap/opencap_clean/web/webviz_markerset/export_osim_markerset_scene.py \
  --model /root/TVB/SAM4Dcap/opencap_clean/output/Data/subject2_Session0/OpenSimData/mmpose_0.8/2-cameras/Model/LaiUhlrich2022_scaled.osim \
  --out web/webviz_markerset/scene_LaiUhlrich2022_scaled.json \
  --marker-suffix _study

cd /root/TVB/SAM4Dcap/opencap_clean
python -m http.server 8090 --bind 127.0.0.1

