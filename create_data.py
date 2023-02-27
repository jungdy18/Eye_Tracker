import cv2
import mediapipe as mp
import mouse
import keyboard
import time
import pandas as pd
from screeninfo import get_monitors
from keras.models import load_model


model = load_model("train_model.h5")

# left_eye = [468,469,470,471,472]
# right_eye = [473,474,475,476,477]

save_data = True
auto_move = False
m_event = False
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)



while True:
	_,frame = cam.read()

	if keyboard.is_pressed("ctrl"):
		auto_move = True
	
	if mouse.is_pressed() and save_data==True:
		m_event = True
	

	if m_event or auto_move:
		eye_dict = {"l_c":468,"l_r":469,"l_t":470,"l_l":471,"l_b":472,"r_c":473,"r_r":474,"r_t":475,"r_l":476,"r_b":477}
		frame = cv2.flip(frame,1)
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		output = face_mesh.process(rgb_frame)
		landmark_points = output.multi_face_landmarks
		frame_h, frame_w, _ = frame.shape
		if landmark_points:
			landmarks = landmark_points[0].landmark
			for k,v in eye_dict.items():
				eye_dict[k] = [round(landmarks[v].x * frame_w,6), round(landmarks[v].y * frame_h,6)]
			eye_dict["l_c_x"] = eye_dict["l_c"][0]
			eye_dict["l_c_y"] = eye_dict.pop("l_c")[1]
			eye_dict["l_r_x"] = eye_dict["l_r"][0]
			eye_dict["l_r_y"] = eye_dict.pop("l_r")[1]
			eye_dict["l_t_x"] = eye_dict["l_t"][0]
			eye_dict["l_t_y"] = eye_dict.pop("l_t")[1]
			eye_dict["l_l_x"] = eye_dict["l_l"][0]
			eye_dict["l_l_y"] = eye_dict.pop("l_l")[1]
			eye_dict["l_b_x"] = eye_dict["l_b"][0]
			eye_dict["l_b_y"] = eye_dict.pop("l_b")[1]
			eye_dict["r_c_x"] = eye_dict["r_c"][0]
			eye_dict["r_c_y"] = eye_dict.pop("r_c")[1]
			eye_dict["r_r_x"] = eye_dict["r_r"][0]
			eye_dict["r_r_y"] = eye_dict.pop("r_r")[1]
			eye_dict["r_t_x"] = eye_dict["r_t"][0]
			eye_dict["r_t_y"] = eye_dict.pop("r_t")[1]
			eye_dict["r_l_x"] = eye_dict["r_l"][0]
			eye_dict["r_l_y"] = eye_dict.pop("r_l")[1]
			eye_dict["r_b_x"] = eye_dict["r_b"][0]
			eye_dict["r_b_y"] = eye_dict.pop("r_b")[1]
			

		if m_event and len((eye_dict))==20:
			eye_dict["x"] = mouse.get_position()[0]
			eye_dict["y"] = mouse.get_position()[1]
			df = pd.DataFrame.from_dict(eye_dict,orient="index").T
			df.to_csv('database.csv', index=False, mode='a', encoding='utf-8-sig', header=False)
			m_event=False
			while(mouse.is_pressed()):
				time.sleep(0.2)
		
		if auto_move and len((eye_dict))==20:
			predict_data = model.predict([list(eye_dict.values())])[0]
			if 0 <= int(predict_data[0]) <= get_monitors()[0].width and 0 <= int(predict_data[1]) <= get_monitors()[0].height :
				mouse.move(int(predict_data[0]),int(predict_data[1]),absolute=True)
				time.sleep(0.3)
			auto_move = False
		

	cv2.waitKey(1)