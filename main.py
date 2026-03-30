import numpy as np
import sys
import pygame
import scipy

pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 670
BACKGROUND_COLOR = (30, 30, 35)
BOID_COLOR = (120, 170, 240)
BOID_COUNT = 5
TARGET_COLOR = (255, 100, 100)
TARGET_RADIUS = 8
FPS = 65


def limit(vector, max_magnitude):
    magnitude = np.linalg.norm(vector)
    if magnitude > max_magnitude:
        return (vector / magnitude) * max_magnitude
    return vector


class Boid:
    def __init__(self):
        self.position = np.array([
            np.random.randint(0, SCREEN_WIDTH),
            np.random.randint(0, SCREEN_HEIGHT),
        ], dtype=float)
        self.velocity = np.random.uniform(-3, 3, 2)
        self.size = 6
        self.max_speed = 4
        self.max_force = 0.3

        self.perception = 50
        self.align_radius = self.perception * 1.5
        self.coh_radius = self.perception * 2

        self.wall_weight = 0.9
        self.align_weight = 1.0
        self.coh_weight = 0.5
        self.sep_weight = 1.0
        self.target_weight = 0.8

    def draw(self, screen):
        speed = np.linalg.norm(self.velocity)
        d = self.velocity / speed if speed > 1e-4 else np.array([1.0, 0.0])
        perp = np.array([-d[1], d[0]])

        tip   = self.position + d    * self.size
        left  = self.position - d    * self.size + perp * self.size
        right = self.position - d    * self.size - perp * self.size

        pygame.draw.polygon(screen, BOID_COLOR, [tip.astype(int), left.astype(int), right.astype(int)])

    def update(self, boids, target, avoid_walls):
        self._apply_target(target)
        self._apply_flocking(boids)
        if avoid_walls:
            self._apply_wall_avoidance()

        self.velocity = limit(self.velocity, self.max_speed)
        self.position += self.velocity
        self.position[0] %= SCREEN_WIDTH
        self.position[1] %= SCREEN_HEIGHT

    def _steer_toward(self, desired):
        """Return a steering force toward a desired velocity vector."""
        mag = np.linalg.norm(desired)
        if mag < 1e-6:
            return np.zeros(2)
        desired = (desired / mag) * self.max_speed
        return limit(desired - self.velocity, self.max_force)

    def _apply_target(self, target):
        to_target = target - self.position
        self.velocity += self._steer_toward(to_target) * self.target_weight

    def _apply_flocking(self, boids):
        sep_sum = np.zeros(2)
        align_sum = np.zeros(2)
        coh_sum = np.zeros(2)
        sep_total = align_total = coh_total = 0

        for b in boids:
            if b is self:
                continue
            diff = self.position - b.position
            d2 = diff[0] ** 2 + diff[1] ** 2
            if d2 == 0:
                continue

            if d2 < self.perception ** 2:
                sep_sum += diff / (d2 + 1e-6)
                sep_total += 1

            if d2 < self.align_radius ** 2:
                align_sum += b.velocity
                align_total += 1

            if d2 < self.coh_radius ** 2:
                coh_sum += b.position
                coh_total += 1

        if sep_total > 0:
            self.velocity += self._steer_toward(sep_sum / sep_total) * self.sep_weight

        if align_total > 0:
            self.velocity += self._steer_toward(align_sum / align_total) * self.align_weight

        if coh_total > 0:
            center = coh_sum / coh_total
            self.velocity += self._steer_toward(center - self.position) * self.coh_weight

    def _apply_wall_avoidance(self):
        margin = self.perception * 0.7
        wall = np.zeros(2)

        if self.position[0] < margin:
            wall[0] += (margin - self.position[0]) / margin
        elif self.position[0] > SCREEN_WIDTH - margin:
            wall[0] -= (self.position[0] - (SCREEN_WIDTH - margin)) / margin

        if self.position[1] < margin:
            wall[1] += (margin - self.position[1]) / margin
        elif self.position[1] > SCREEN_HEIGHT - margin:
            wall[1] -= (self.position[1] - (SCREEN_HEIGHT - margin)) / margin

        self.velocity += self._steer_toward(wall) * self.wall_weight


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

boids = [Boid() for _ in range(BOID_COUNT)]
mouse_down = False
mouse_target = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False

    screen.fill(BACKGROUND_COLOR)

    if mouse_down:
        mouse_target = np.array(pygame.mouse.get_pos(), dtype=float)
        pygame.draw.circle(screen, TARGET_COLOR, mouse_target.astype(int), TARGET_RADIUS)

    for b in boids:
        b.update(boids, mouse_target, False)
        b.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)