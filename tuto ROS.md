
- Si jamais la commande `roscd beginner_tutorials` renvoit
``` 
roscd: No such package/stack 'beginner_tutorials'
```

Faire:
```
cd ~/catkin_ws
source devel/setup.bash
```


## Se connecter aux turtles

```bash
ssh burger1
```
- Remplacer `192.168.1.216` par l'ip perso trouvée sur `ip a`
```bash
export ROS_MASTER_URI=http://192.168.1.216:11311/
```

```bash
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```

- Sur le pc perso (pas en ssh)

```bash
source ~\.bashrc
roscore
```

## Simuler une turtle

```bash
roscore
```

```bash
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```



## Controler la turtle

- Dans un autre terminal
```bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

## Voir le LIDAR

```bash
roslaunch turtlebot3_slam turtlebot3_slam.launch
```

# Les F1Tenth

## Lancer la simulation

```bash
roslaunch f1tenth_simulator simulator.launch
```

## Lancer ses scripts perso

- Penser à les ajouter dans le CMakeList.txt avant.

```bash
rosrun f1tenth_perso emergency_breaking.py
```
