# maritime-collision-avoidance-thesis

## Objective
To develop a collision avoidance software system for maritime environments. 

## Participants
The project is being undertaken in a team of three undergraduates, Taj Paramanad-Wilson, Brent Morgan and Brandon Falconerunder the academic supervision of Simon Denman. <br>
<br>

### Object Tracking Specifications  
  
A camera sensor attached to a naval vessel, at approximately 10-20cm above the waterline, will be used to provide a stream of video footage to the model. The system needs to be able to detect moving and stationary objects in a marine environment, in real-time. The model must be able to maintain a high tracking accuracy during all-weather scenarios, like during storms and low light scenes. Additionally, the system must be able to track both objects on the horizon and in the foreground of a scene. To account for object occlusion, and objects re-entering the scene, a unique identifier will need to be placed on all objects. The accuracy and performance of the model is of great importance, as the system will be implemented using embedded hardware with processing constraints. 

The system would be one that relied on a correlation filter to determine the future position of an object over time. This correlation filter would likely be one based on the MOSSE filter. A motion model would be implemented with the standard Kalman filter. 

The input of the system is a stream of video data, and the initial positions of all the objects in a scene via object detection. The output of the system are the coordinates of a bounding box for object’s track, along with a unique identifier. 

#### Requirements
1.	The system must be able to track multiple objects, both in the foreground and on the horizon of a maritime scene.
2.	The system must be able to track objects in real-time (Above 30fps).
3.	The system should be able to have high accuracy (Above 0.6 IoU) in all scenarios, under different weather conditions.
4.	The system must be able to track up to 10 different objects at once, with minimal false positive track predictions.
5.	The system must keep a track of all objects with a unique identifier, and maintain this identifier when the object is off screen and occluded. 
