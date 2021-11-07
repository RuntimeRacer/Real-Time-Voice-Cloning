build:
	docker build -t voice-clone:latest .

run:
	docker run -it --rm \
		-v "$(PWD)":/workspace \
		-v /media/sberryman/nvme_1tb/speech/sv2tts:/output \
		-v /media/sberryman/tv_media2/datasets:/datasets \
		voice-clone:latest

train_cv:
	docker run -d --rm \
		-v "$(PWD)":/workspace \
		-v /media/sberryman/nvme_1tb/speech/sv2tts:/output \
		-v /media/sberryman/tv_media2/datasets:/datasets_slr \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		voice-clone:latest \
		python \
		encoder_train.py \
		cv_run \
		/output/encoder \
		--models_dir=/output/models \
		--visdom_server="http://192.168.7.171"

train_mixed:
	docker run -d --rm \
		-v "$(PWD)":/workspace \
		-v /media/sberryman/nvme_1tb/speech/sv2tts:/output \
		-v /media/sberryman/tv_media2/datasets:/datasets_slr \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		voice-clone:latest \
		python \
		encoder_train.py \
		mixed_run \
		/output/encoder \
		--models_dir=/output/models \
		--visdom_server="http://192.168.7.171"

train_english:
	docker run -d --rm \
		-v "$(PWD)":/workspace \
		-v /media/sberryman/nvme_1tb/speech/sv2tts_english:/output \
		-v /media/sberryman/tv_media2/datasets:/datasets_slr \
		-e NVIDIA_VISIBLE_DEVICES=1 \
		voice-clone:latest \
		python \
		encoder_train.py \
		english_run \
		/output/encoder \
		--models_dir=/output/models \
		--visdom_server="http://192.168.7.171"

train_english_minibeast:
	docker run -d --rm \
		-v /mnt/ssd_1tb/speech/sv2tts_english/encoder:/output/encoder \
		-v /mnt/nvme_1tb/speech/sv2tts_english/models:/output/models \
		-v /media/sberryman/tv_media2/datasets:/datasets_slr \
		voice-clone:latest \
		python \
		encoder_train.py \
		english_run \
		/output/encoder \
		--models_dir=/output/models \
		--visdom_server="http://192.168.7.171"

visdom:
	docker run -d --rm \
		-p 8097:8097 \
		-v "$(PWD)":/workspace \
		-v  /media/sberryman/nvme_1tb/speech/sv2tts:/output \
		voice-clone:latest \
		visdom
	echo "Vizdom now available at http://192.168.7.171:8097"

preprocess:
	python encoder_preprocess.py \
		/datasets \
		--out_dir=/output/encoder \
		--datasets="voxceleb2" \
		--skip_existing

train:
	python encoder_train.py \
		cv_run \
		/output/encoder \
		--models_dir=/output/models \
		--visdom_server="http://192.168.7.171"
