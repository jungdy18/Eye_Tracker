import cv2
import mediapipe as mp
import mouse
import keyboard
import time

# left_eye = [468,469,470,471,472]
# right_eye = [473,474,475,476,477]

save_data = True
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)

auto_move = False

while cam.isOpened():
	_,frame = cam.read()

	if keyboard.is_pressed("ctrl"):
		auto_move = True

	if save_data or auto_move:
		eye_dict = {"l_c":468,"l_r":469,"l_t":470,"l_l":471,"l_b":472,"r_c":473,"r_r":474,"r_t":475,"r_l":476,"r_b":477}
		frame = cv2.flip(frame,1)
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		output = face_mesh.process(rgb_frame)
		landmark_points = output.multi_face_landmarks
		frame_h, frame_w, _ = frame.shape
		if landmark_points:
			landmarks = landmark_points[0].landmark
			for k,v in eye_dict.items():
				eye_dict[k] = [landmarks[v].x * frame_w, landmarks[v].y * frame_h]
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
			eye_dict["x"] = mouse.get_position()[0]
			eye_dict["y"] = mouse.get_position()[1]

		print(eye_dict)
		auto_move = False
		time.sleep(0.3)
	cv2.waitKey(1)





# -----------------------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp
import pyautogui as pg

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)

# left_eye = [468,469,470,471,472]
# right_eye = [473,474,475,476,477]



while True:
	eye_dict = {"left_ceter":468,"left_right":469,"left_top":470,"left_left":471,"left_bottom":472,"right_center":473,"right_right":474,"right_top":475,"right_left":476,"right_bottom":477}
	_,frame = cam.read()
	frame = cv2.flip(frame,1)
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	output = face_mesh.process(rgb_frame)
	landmark_points = output.multi_face_landmarks
	frame_h, frame_w, _ = frame.shape
	if landmark_points:
		landmarks = landmark_points[0].landmark
		for k,v in eye_dict.items():
			eye_dict[k] = [landmarks[v].x * frame_w, landmarks[v].y * frame_h]
		print(eye_dict)
	cv2.imshow("eye",frame)
	cv2.waitKey(1)



# ----------------------------------------------------------------------------------------

import cv2
import mediapipe as mp
import pyautogui as pg

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)

left_eye = [468,469,470,471,472]
right_eye = [473,474,475,476,477]
eye_dict = {"left_ceter":468,"left_right":469,"left_top":470,"left_left":471,"left_bottom":472,"right_center":473,"right_right":474,"right_top":475,"right_left":476,"right_bottom":477}



while True:
	eye_landmarks = []
	_,frame = cam.read()
	frame = cv2.flip(frame,1)
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	output = face_mesh.process(rgb_frame)
	landmark_points = output.multi_face_landmarks
	frame_h, frame_w, _ = frame.shape
	if landmark_points:
		landmarks = landmark_points[0].landmark
		for i in left_eye + right_eye:
			eye_landmarks.append(landmarks[i])
		for landmark in eye_landmarks:
			# (x,y)  = (landmark.x * frame_w, landmark.y * frame_h)
			x = int(landmark.x * frame_w)
			y = int(landmark.y * frame_h)
			cv2.circle(frame, (x,y),1,(0,255,0))
			print(x,y)
	cv2.imshow("eye",frame)
	cv2.waitKey(1)
