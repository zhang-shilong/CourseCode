import random
import itertools
import pygame
import time
import copy
import os

directions_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


class Cell:

    def __init__(self, telomere_length=50):
        # genome factor
        self.self_growth = False  # 自我生长，不需要生长的外部信号即可生长
        self.ignore_growth_inhabit = False  # 忽略生长抑制，临近细胞发出抗生长信号时仍可生长
        self.evasion_of_apoptosis = False  # 规避细胞凋亡，高水平遗传损伤下不死亡
        self.immortality = False  # 永生，忽略端粒长度的影响
        self.total_flag = 0  # 以上标志为True的数量，有任一一个True即判定为癌变细胞
        # parameters
        self.genetic_instability = 0.001  # 遗传不稳定性
        self.telomere_length = telomere_length  # 端粒长度
        self.mitosis_probability = 0  # 有丝分裂的概率，随回合变化
        self.rounds_before_mitosis = 10  # 距离必须发生有丝分裂的回合数
        self.current_mitosis_rounds = self.rounds_before_mitosis  # 距离必须发生有丝分裂的回合数，变量
        self.mitosis_round_minimum = 5  # 每次分裂后的最小等待回合数
        self.mitosis_probability_increment = \
            1 / (self.rounds_before_mitosis - self.mitosis_round_minimum)  # 每回合增加该细胞有丝分裂的概率
        self.random_death_probability = 0.001  # 每个细胞都有一定概率死亡
        self.apoptosis_probability = 10  # 细胞凋亡的概率：total_flag / apoptosis_probability
        self.kill_neighbor_probability = 0.03  # 已获得ignore_growth_inhabit的细胞杀死隔壁细胞的概率

    def is_cancerous_cell(self):
        return bool(self.total_flag)

    def genetic_mutation(self, factor):
        tmp = random.random()
        flag = tmp < self.genetic_instability
        if flag:
            if factor:
                self.total_flag -= 1
            else:
                self.total_flag += 1
        return flag

    def genetic_mutation_all(self):
        if not self.self_growth and self.genetic_mutation(self.self_growth):
            self.self_growth = not self.self_growth
        if not self.ignore_growth_inhabit and self.genetic_mutation(self.ignore_growth_inhabit):
            self.ignore_growth_inhabit = not self.ignore_growth_inhabit
        if not self.evasion_of_apoptosis and self.genetic_mutation(self.evasion_of_apoptosis):
            self.evasion_of_apoptosis = not self.evasion_of_apoptosis
        if not self.immortality and self.genetic_mutation(self.immortality):
            self.immortality = not self.immortality

    def check_mitosis(self, in_growth_factor_area, has_space):
        if not has_space:
            return False
        if in_growth_factor_area or self.self_growth:
            if self.current_mitosis_rounds == 0:
                return True
            if self.current_mitosis_rounds <= self.mitosis_round_minimum:
                if random.random() < self.mitosis_probability:
                    return True
                self.mitosis_probability += self.mitosis_probability_increment
            return False

    def check_random_death(self):
        if random.random() < self.random_death_probability:
            return True
        return False

    def check_apoptosis(self):
        if self.evasion_of_apoptosis:
            return False
        if random.random() < self.total_flag / self.apoptosis_probability:
            return True
        return False

    def check_kill_neighbor(self):
        if self.ignore_growth_inhabit and random.random() < self.kill_neighbor_probability:
            return True
        return False

    def check_treatment_death(self, lethality_normal, lethality_cancerous):
        tmp = random.random()
        if self.is_cancerous_cell():
            if tmp < lethality_cancerous:
                return True
        else:
            if tmp < lethality_normal:
                return True
        return False


class CellularAutomata:

    def __init__(self, do_intermittent_treatment=False, do_continuous_treatment=False):
        if do_intermittent_treatment and do_continuous_treatment:
            raise Exception("Please choose one type of treatment")
        # basic settings and parameters
        self.board = 120
        self.cells = [[None for _ in range(self.board)] for _ in range(self.board)]
        self.growth_factor_ratio = 0.8
        self.growth_factor_area_minimum = int(self.board * (1 - self.growth_factor_ratio))
        self.growth_factor_area_maximum = int(self.board * self.growth_factor_ratio)
        self.total_normal_cells = 0
        self.total_cancerous_cells = 0
        self.total_cells = 0
        # treatment settings
        self.do_intermittent_treatment = do_intermittent_treatment  # 是否采取间歇治疗
        self.do_continuous_treatment = do_continuous_treatment  # 是否采取连续治疗
        # intermittent treatment
        self.rounds_before_treatment = 10  # 距离间歇治疗的回合数，常量
        self.current_treatment_rounds = self.rounds_before_treatment  # 距离间接治疗的回合数，变量
        self.intermittent_lethality_normal = 0.1
        self.intermittent_lethality_cancerous = 0.5
        # continuous treatment
        self.continuous_lethality_normal = 0.01
        self.continuous_lethality_cancerous = 0.05

    def add_cell(self, x, y, cell):
        self.total_cells += 1
        if cell.is_cancerous_cell():
            self.total_cancerous_cells += 1
        else:
            self.total_cancerous_cells += 1
        self.cells[x][y] = cell

    def remove_cell(self, x, y):
        cell = self.cells[x][y]
        if not cell:
            return
        self.total_cells -= 1
        if cell.is_cancerous_cell():
            self.total_cancerous_cells += 1
        else:
            self.total_cancerous_cells += 1
        self.cells[x][y] = None

    def initialization(self):
        tmp = self.board // 2
        for x, y in itertools.product(range(tmp - 10, tmp + 11), range(tmp - 10, tmp + 11)):
            self.add_cell(x, y, Cell())

    def check_in_board(self, x, y):
        return 0 <= x < self.board and 0 <= y < self.board

    def check_growth_area(self, x, y):
        if self.growth_factor_area_minimum <= x <= self.growth_factor_area_maximum and \
                self.growth_factor_area_minimum <= y <= self.growth_factor_area_maximum:
            return True
        return False

    def check_has_space(self, x, y):
        for dx, dy in directions_list:
            new_x = x + dx
            new_y = y + dy
            if self.check_in_board(new_x, new_y) and not self.cells[new_x][new_y]:
                return True
        return False

    def mitosis(self, x, y):
        random.shuffle(directions_list)
        for dx, dy in directions_list:
            new_x = x + dx
            new_y = y + dy
            if self.check_in_board(new_x, new_y) and not self.cells[new_x][new_y]:
                self.cells[x][y].telomere_length -= 1
                self.cells[x][y].mitosis_probability = 0
                self.cells[x][y].current_mitosis_rounds = self.cells[x][y].rounds_before_mitosis
                if self.cells[x][y].telomere_length < 0 and not self.cells[x][y].immortality:
                    self.remove_cell(x, y)
                else:
                    self.cells[new_x][new_y] = copy.copy(self.cells[x][y])
                return

    def kill_neighbor(self, x, y):
        random.shuffle(directions_list)
        for dx, dy in directions_list:
            new_x = x + dx
            new_y = y + dy
            if self.check_in_board(new_x, new_y) and self.cells[new_x][new_y] and \
                    not self.cells[new_x][new_y].is_cancerous_cell():
                self.remove_cell(new_x, new_y)
                return

    def run_one_round(self):
        for x, y in itertools.product(range(self.board), range(self.board)):
            cell = self.cells[x][y]
            if not cell:
                continue
            if cell.check_random_death():
                self.remove_cell(x, y)
                continue
            if cell.check_apoptosis():
                self.remove_cell(x, y)
                continue
            cell.genetic_mutation_all()
            if cell.check_mitosis(self.check_growth_area(x, y), self.check_has_space(x, y)):
                self.mitosis(x, y)
                if cell.check_kill_neighbor():
                    self.kill_neighbor(x, y)
            cell.current_mitosis_rounds -= 1

            if self.do_continuous_treatment and \
                    cell.check_treatment_death(self.continuous_lethality_normal,
                                               self.continuous_lethality_cancerous):
                self.remove_cell(x, y)

            if self.do_intermittent_treatment and self.current_treatment_rounds == 0 and \
                    cell.check_treatment_death(self.intermittent_lethality_normal,
                                               self.intermittent_lethality_cancerous):
                self.remove_cell(x, y)

    def pygame_visualization(self, pixels_per_cell=4, run_name="default"):
        path = "images/" + run_name
        if not os.path.exists(path):
            os.makedirs(path)
        pygame.init()
        tmp = self.board * pixels_per_cell
        screen = pygame.display.set_mode((tmp, tmp))
        pygame.display.set_caption("Cellular Automata")
        screen.fill((255, 255, 255))

        self.initialization()
        keep_going = True
        run_times = 0
        while keep_going:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    keep_going = False
            for x, y in itertools.product(range(self.board), range(self.board)):
                cell = self.cells[x][y]
                if not cell:
                    continue
                cell_pos_x = x * pixels_per_cell
                cell_pos_y = y * pixels_per_cell
                if cell.is_cancerous_cell():
                    if cell.total_flag == 4:
                        pygame.draw.rect(screen, (128, 0, 128),
                                         [cell_pos_x, cell_pos_y, pixels_per_cell, pixels_per_cell], 0)
                    else:
                        pygame.draw.rect(screen, (255, 0, 0),
                                         [cell_pos_x, cell_pos_y, pixels_per_cell, pixels_per_cell], 0)
                else:
                    pygame.draw.rect(screen, (0, 255, 0),
                                     [cell_pos_x, cell_pos_y, pixels_per_cell, pixels_per_cell], 0)
            pygame.display.update()
            run_times += 1
            self.run_one_round()
            if self.current_treatment_rounds == 0:
                self.current_treatment_rounds = self.rounds_before_treatment
            self.current_treatment_rounds -= 1
            if run_times % 100 == 0:
                file_name = path + "/ca-" + str(run_times) + ".png"
                pygame.image.save(screen, file_name)
            if run_times == 900:
                keep_going = False
            time.sleep(0.01)


if __name__ == "__main__":
    # default (no treatment)
    ca1 = CellularAutomata()
    ca1.pygame_visualization()
    # continuous treatment
    ca2 = CellularAutomata(do_continuous_treatment=True)
    ca2.pygame_visualization(run_name="continuous_treatment")
    # intermittent treatment
    ca3 = CellularAutomata(do_intermittent_treatment=True)
    ca3.pygame_visualization(run_name="intermittent_treatment")
