import cv2
import mediapipe as mp
import numpy as np
import math as math
import time


angle_list = []
squat_count = 0

def squat_angle(r_ankle, r_hip, r_knee):
  v1 = [r_hip.x- r_knee.x, r_hip.y- r_knee.y, r_hip.z- r_knee.z]
  v2 = [r_ankle.x- r_knee.x, r_ankle.y- r_knee.y, r_ankle.z- r_knee.z]

  v1mag = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
  v1norm = [v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag]

  v2mag = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
  v2norm = [v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag]

  res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]

  angle = math.acos(res) * (180/3.14)

  return angle

def is_Squat(angle):
  squat_bool = False
  if(angle < 85):
    squat_bool = True
  return squat_bool



if __name__ == '__main__':

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose()

  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      results = pose.process(image)
      

      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

      cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

      if (results.pose_world_landmarks):
          right_knee = results.pose_world_landmarks.landmark[26] 
          right_hip = results.pose_world_landmarks.landmark[24]
          right_ankle = results.pose_world_landmarks.landmark[28]
          angle = squat_angle(right_ankle, right_hip, right_knee)
          angle_list.append(angle)
          isSquat = is_Squat(angle)
          if isSquat:
            print("Successful Squat")

      if cv2.waitKey(5) & 0xFF == 27:
        break
  
  def count_squats(list):
      squat_count = 0
      in_squat = False
      for angle in list:
          if angle < 85:
              if not in_squat:
                  squat_count += 1
                  in_squat = True
          else:
              in_squat = False

      return squat_count
  reps = count_squats(angle_list)   
  print(f'you completed {reps} squat reps')
  
  cap.release()
  
    