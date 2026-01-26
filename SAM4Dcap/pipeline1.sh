source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/TVB/envs/body4d

python /root/TVB/sam-body4d/scripts/run_batch.py --input /root/TVB/SAM4Dcap/pipeline1/subject2_DJ1_s0_cam1.mp4 --output /root/TVB/SAM4Dcap/pipeline1

mkdir -p /root/TVB/SAM4Dcap/pipeline1/smpl_results

cd /root/TVB/MHR/tools/mhr_smpl_conversion

/root/TVB/envs/MHRtoSMPL/bin/python /root/TVB/MHRtoSMPL/convert_mhr_to_smpl.py \
  --input /root/TVB/SAM4Dcap/pipeline1/mhr_params \
  --output /root/TVB/SAM4Dcap/pipeline1/smpl_results

cd /root/TVB/SAM4Dcap/align
python generate_aligned_trc.py

cp /root/TVB/SMPL2AddBiomechanics/models/bsm/bsm.osim /root/TVB/SAM4Dcap/pipeline1/
cp /root/TVB/SAM4Dcap/pipeline1/bsm.osim /root/TVB/SAM4Dcap/pipeline1/unscaled_generic.osim
mkdir -p /root/TVB/SAM4Dcap/pipeline1/trials/trial1
cp /root/TVB/SAM4Dcap/align/output/aligned_subject2_motion.trc /root/TVB/SAM4Dcap/pipeline1/trials/trial1/markers.trc

conda activate /root/TVB/envs/opensim
cd /root/TVB/AddBiomechanics
python server/engine/src/engine.py /root/TVB/SAM4Dcap/pipeline1 | tee /root/TVB/SAM4Dcap/pipeline1/run.log

cd /root/TVB/AddBiomechanics/frontend/public
ln -sfn /root/TVB/SAM4Dcap/pipeline1 localdata
cd /root/TVB/AddBiomechanics/frontend
HOST=0.0.0.0 PORT=3088 REACT_APP_LOCAL_PREVIEW_URL=/localdata/trials/trial1/segment_1/preview.bin yarn start
