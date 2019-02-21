import cfg
import numpy as np


def clip_norm(vec, clip=1):
    norm = np.linalg.norm(vec)
    if norm > clip:
        vec /= norm
    return vec


class Boid:
    """This class describes a Boids agent."""

    def __init__(self, pos=np.zeros(2), vel=np.zeros(2), acc=np.zeros(2), boost=1, attract=True):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.boost = 1
        self.behavior_type = 1 if attract else -1 # 1 likes to group together, -1 stays away

    def step(self, step_size=cfg.STEP_SIZE):
        self.pos += step_size*self.boost*self.vel

    def get(self, attribute):
        return getattr(self, attribute)

    def get_neighbors(self, boids, radius):
        neighbors = BoidList()
        for other in boids:
            dist = np.linalg.norm(other.pos-self.pos)
            if dist > 0 and dist <= radius:
                neighbors.add(other)
        return neighbors

    def get_cohesion_accel(self, group):
        centroid = np.mean(group.get_attributes('pos'), axis=0)
        accel = centroid-self.pos
        return self.behavior_type*clip_norm(accel)

    def get_alignment_accel(self, group):
        avg_vel = np.mean(group.get_attributes('vel'), axis=0)
        return clip_norm(avg_vel)

    def get_separation_accel(self, group):
        centroid = np.mean(group.get_attributes('pos'), axis=0)
        accel = self.pos-centroid
        return clip_norm(accel)


class BoidList:
    """This class handles the interactions between multiple Boids agents."""

    R_CLOSE = cfg.R_CLOSE
    R_LOCAL = cfg.R_LOCAL

    C_COH = cfg.C_COH
    C_SEP = cfg.C_SEP
    C_ALI = cfg.C_ALI
    C_ORI = cfg.C_ORI

    BOOST_MIN = cfg.BOOST_MIN
    BOOST_MAX = cfg.BOOST_MAX

    BOUND = cfg.BOUND

    def __init__(self, n=0):
        self.boids = []
        self.n = n

        if self.n > 0:
            self.boids = [Boid(pos=np.random.randn(2), vel=np.random.randn(2)) for _ in range(self.n)]
    
    def __getitem__(self, index):
        return self.boids[index]

    def __len__(self):
        return self.n

    def add(self, boid):
        self.boids.append(boid)
        self.n += 1

    def get_attributes(self, attribute):
        ret = []
        for boid in self.boids:
            ret.append(boid.get(attribute))
        return np.array(ret)

    def step(self):
        for boid in self.boids:
            boid.step()

    def update_velocities(self):
        for boid in self.boids:
            local_group = boid.get_neighbors(self.boids, self.R_LOCAL)
            if len(local_group) > 0:
                boid.acc += self.C_COH*boid.get_cohesion_accel(local_group)
                boid.acc += self.C_ALI*boid.get_alignment_accel(local_group)

            close_group = boid.get_neighbors(self.boids, self.R_CLOSE)
            if len(close_group) > 0:
                boid.acc += self.C_SEP*boid.get_separation_accel(close_group)

            dist = np.linalg.norm(boid.pos)
            if dist > self.BOUND:
                boid.acc += self.C_ORI*clip_norm(-boid.pos)

        for boid in self.boids:
            boid.vel += 0.5*clip_norm(boid.acc)
            boid.vel = clip_norm(boid.vel)
            boid.acc = np.zeros(2)

    def sample_boost(self):
        for boid in self.boids:
            boid.boost = np.random.uniform(self.BOOST_MIN, self.BOOST_MAX)
