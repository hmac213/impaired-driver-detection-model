import math
import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(
        self,
        road,
        impairment_level=0,
        mass=3000,
        max_lateral_force=10000
    ):
        self.start_point = road.start_point
        self.goal_point = road.end_point
        self.speed = road.speed_limit

        self.impairment_level = impairment_level # range from 0 to 1
        self.road = road
        self.mass = mass
        self.max_lateral_force = max_lateral_force
        # add initial noise to position scaled by impairment
        noise_range = 0.1 + self.impairment_level * 0.2
        noise_x = random.uniform(-noise_range, noise_range)
        noise_y = random.uniform(-noise_range, noise_range)
        self.position = (self.start_point[0] + noise_x,
                         self.start_point[1] + noise_y)
        # record trajectory for plotting
        self.history = [self.position]
        # initial heading: straight ahead along +x axis
        self.heading = 0.0
        # steering interruption state
        self.steering_interrupted = False
        self.steering_rate = 0.0
        self.interrupt_timer = 0.0
        self.reaction_time = 0.0

    def calc_lateral_foce(self, speed_mph, turn_radius_ft):
        speed_ft_per_sec = speed_mph * 5280 / 3600
        lateral_accel = speed_ft_per_sec ** 2 / turn_radius_ft
        lateral_force = self.mass * lateral_accel

        return lateral_force

    def update_heading(self, dt):
        """
        Handle random steering interruptions and corrections based on impairment level.
        """
        # discrete steering jerks determined by impairment
        if not getattr(self, 'jerking', False):
            # chance to start a jerk: higher if more impaired
            half_width = self.road.width / 2
            distance_factor = max(0.0, 1.0 - abs(self.position[1]) / half_width)
            if random.random() < self.impairment_level * 0.2 * distance_factor * dt:
                # compute max angular rate from lateral force limit
                speed_ft_s = self.speed * 5280 / 3600
                lateral_accel_max = self.max_lateral_force / self.mass
                base_rate = lateral_accel_max / speed_ft_s
                # jerk intensity scaled by impairment and noise
                jerk_intensity = random.uniform(-base_rate, base_rate) * (0.5 + self.impairment_level)
                self.jerking = True
                self.jerk_rate = jerk_intensity
                # jerk duration scaled by impairment
                self.jerk_duration = random.uniform(0.1, 0.3) * (1 + self.impairment_level)
                self.jerk_timer = 0.0
        else:
            # apply jerk until duration elapses
            self.jerk_timer += dt
            if self.jerk_timer < self.jerk_duration:
                self.heading += self.jerk_rate * dt
            else:
                self.jerking = False

        # if no interruption, possibly start one when roughly pointing at goal
        if not self.steering_interrupted:
            # include baseline interruptions for all drivers
            if random.random() < (0.1 + self.impairment_level * 0.5) * dt:
                # compute max angular rate from lateral force limit
                speed_ft_s = self.speed * 5280 / 3600
                lateral_accel_max = self.max_lateral_force / self.mass
                max_rate = lateral_accel_max / speed_ft_s
                # random steering drift rate scaled by impairment
                self.steering_rate = random.uniform(-max_rate, max_rate) * (0.5 + self.impairment_level)
                # reaction time longer for higher impairment
                self.reaction_time = random.uniform(0.5, 1.5) * (1 + self.impairment_level)
                self.interrupt_timer = 0.0
                self.steering_interrupted = True
        else:
            # during interruption
            self.interrupt_timer += dt
            if self.interrupt_timer < self.reaction_time:
                # drift heading
                self.heading += self.steering_rate * dt
            else:
                # correct heading toward goal
                target_heading = math.atan2(self.goal_point[1] - self.position[1],
                                            self.goal_point[0] - self.position[0])
                # compute smallest angle difference
                error = (target_heading - self.heading + math.pi) % (2 * math.pi) - math.pi
                # compute base max angular rate from lateral force limit
                speed_ft_s = self.speed * 5280 / 3600
                lateral_accel_max = self.max_lateral_force / self.mass
                base_rate = lateral_accel_max / speed_ft_s
                # determine correction rate: at edge use full force, otherwise scale by impairment
                y = self.position[1]
                if abs(y) > self.road.width / 2:
                    correction_rate = base_rate
                else:
                    correction_rate = base_rate * (1 + self.impairment_level)
                # apply correction limited by force limit (ensures turning radius >= min radius)
                delta = max(-correction_rate * dt, min(error, correction_rate * dt))
                self.heading += delta
                # end interruption when aligned
                if abs(error) < 0.01:
                    self.steering_interrupted = False
        
        # immediate pull-back when outside lane
        half_width = self.road.width / 2
        y = self.position[1]
        if abs(y) > half_width:
            # compute base correction rate
            speed_ft_s = self.speed * 5280 / 3600
            lateral_accel_max = self.max_lateral_force / self.mass
            base_rate = lateral_accel_max / speed_ft_s
            correction_rate = base_rate * (2 + self.impairment_level)  # stronger pull-back
            # apply correction toward center
            self.heading += math.copysign(correction_rate * dt, -y)

    def move_towards_goal(self, dt):
        # move in direction of current heading
        speed_ft_s = self.speed * 5280 / 3600
        move_dist = speed_ft_s * dt
        dx = math.cos(self.heading) * move_dist
        dy = math.sin(self.heading) * move_dist
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        # clamp x to goal and y within some range (optional)
        if new_x >= self.goal_point[0]:
            new_x = self.goal_point[0]
        self.position = (new_x, new_y)
        self.history.append(self.position)

    def run(self, dt=1/30):
        """
        Run simulation without real-time delays, return trajectory.
        """
        while self.position[0] < self.goal_point[0]:
            self.update_heading(dt)
            self.move_towards_goal(dt)
        return self.history

# all measures in feet
class Road:
    def __init__(
        self,
        width=12,
        speed_limit=45,
    ):
        self.width = width
        self.speed_limit = speed_limit
        self.length = self.calc_length()

        self.start_point = (0, 0)
        self.end_point = (self.length, 0)
        
    def calc_length(self):
        speed_ft_per_sec = self.speed_limit * 5280 / 3600
        return speed_ft_per_sec * 5
    
road = Road()
plt.figure()
for i in range(1):
    agent = Agent(road, impairment_level=1)
    history = agent.run()
    xs, ys = zip(*history)
    plt.plot(xs, ys)
plt.xlabel('X position (ft)')
plt.ylabel('Y position (ft)')
plt.title('Multiple Agent Trajectories')
plt.axhline(road.width / 2, linestyle='--')
plt.axhline(-road.width / 2, linestyle='--')
plt.axis('equal')
plt.show()
