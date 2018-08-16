SET SF=data/audios/00001.wav
rm -rf input.mp4
rm -rf output.mp4
activate tensorflow
python run.py --sf %SF%
python pix2pix.py --mode test --output_dir test_output --input_dir testing_output_images --checkpoint checkpoints/output/

ffmpeg -r 30 -f image2 -s 256x256 -i test_output/images/%d-outputs.png -vcodec libx264 -crf 25 output0.mp4
ffmpeg -r 30 -f image2 -s 256x256 -i test_output/images/%d-inputs.png -vcodec libx264 -crf 25 input0.mp4
ffmpeg -i %SF% output_audio_trim.wav

ffmpeg -i output0.mp4 -i output_audio_trim.wav -c:v copy -c:a aac -strict experimental output.mp4
ffmpeg -i input0.mp4 -i output_audio_trim.wav -c:v copy -c:a aac -strict experimental input.mp4

rm -rf testing_output_images
rm -rf test_output
rm -rf output0.mp4
rm -rf input0.mp4
rm -rf output_audio_trim.wav
rm -rf input.mp4