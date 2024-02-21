import random as rdm
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from random import random, choice

import networkx as nx
from prettytable import PrettyTable

class Node(ABC):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_position(self) -> tuple:
        return (self.x, self.y)
    #implemetira se za svako polje koje nasledjuje klasu drugacije tako da za sta stavljamo pass
    @abstractmethod
    def get_reward(self) -> float:
        pass

    def is_steppable(self) -> bool:
        return True

    def is_terminal(self) -> bool:
        return False

    def has_value(self) -> bool:
        return True
    
class RegularNode(Node):
    #ima neku nagradu
    def __init__(self, reward: float, x: int, y: int):
        super().__init__(x, y)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward
    
class TerminalNode(Node):
    def __init__(self, reward: float, x: int, y: int):
        super().__init__(x, y)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward

    def is_terminal(self) -> bool:
        return True
    #nema vrednost
    def has_value(self) -> bool:
        return False

class WallNode(Node):
    def __init__(self, x: int, y: int):
        super().__init__(x, y)

    def get_reward(self) -> float:
        return 0

    def is_steppable(self) -> bool:
        return False

    def has_value(self) -> bool:
        return False

class TeleportNode(Node):
    def __init__(self, reward: float, x: int, y: int):
        super().__init__(x, y)
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward

ACTIONS = [1, 2, 3]

class MazeEnvironment:
    def __init__(self, dimensions: tuple[int, int]):
        #inicijalizujemo velicinu lavirinta
        self.graph_height = dimensions[0]
        self.graph_width = dimensions[1]
        self.graph = self.initialize_graph(self.graph_height, self.graph_width)

    def initialize_graph(self, width: int, height: int) -> dict:
        #inicijalizujemo graf koji predstavlja lavirint
        graph = {}
        terminal_node_created = False

        for w in range(1, width + 1):
            for h in range(1, height + 1):
                node = self.generate_random_node(w, h)
                graph[node] = []
                #da li imamo terminalni cvor
                if isinstance(node, TerminalNode):
                    terminal_node_created = True

        #ako nemamo nas zadatak uospte tema smisla tako da ga stavljamo na randmo mesto, umesto nekog cvora koji NIJE zid
        if not terminal_node_created:
            non_wall_nodes = [n for n in graph.keys() if not isinstance(n, WallNode)]
            node_to_replace = choice(non_wall_nodes)
            terminal_node = TerminalNode(-1, node_to_replace.x, node_to_replace.y)
            graph.pop(node_to_replace)
            graph[terminal_node] = []

        #postavljamo verovatnoca prelaska iz jednog u drugi cvor
        for node in graph:
            graph[node] = self.set_probabilities(node, graph)

        return graph

    def set_probabilities(self, node: Node, graph: dict) -> list:
        #postavljamo verovatnoce prelaska, ali postujemo odredjena pravila
        #1. cvorovi tipa Wall i Terminal nemaju verovatnoce prelaska u neki drugi cvor, dok se za Regular i Teleport imaju
        #2. iako je stohasticki model, ne zelimo da moze da se skoci u bas bilo koji cvor zato za pola covrova dajemo 0 sansu prelaska u njih


        #Terminal i Wall node nemaju verovatnoce
        if isinstance(node, (WallNode, TerminalNode)):
            return []
        #za ostale koji nisu zid
        nodes_list = [n for n in graph if not isinstance(n, WallNode)]
        probabilities = []

        #racunamo broj funkcionalnih cvorova koje ce imati nultu verovatnocu
        total_cells = self.graph_width * self.graph_height
        #minimalna vrednost od broja cvorova koji nisu Wall i 0.5 ukupnog broja cvorova, da ne bi mogao da ode u bilo koji drugi cvor
        zero_cells = min(len(nodes_list), total_cells // 2)

        #dodeljujemo nultu verovatnocu cvorovima koji nisu Teleport, a ni Wall jer su oni vec van nodes_list
        if not isinstance(node, TeleportNode):
            for _ in range(zero_cells):
                random_node = rdm.choice(nodes_list)
                nodes_list.remove(random_node)
                probabilities.append([0, random_node, 0])

        # rasporedli ostale cvorove akcijama
        for action in ACTIONS:
            if nodes_list:
                #broj cvorova za dodelu svakoj akciji, mora biti bar 1
                action_len = len(nodes_list) // len(ACTIONS) or 1
                nodes_for_action = nodes_list[:action_len]
                nodes_list = nodes_list[action_len:]

                non_zero_probabilities = self.generate_probabilities(nodes_for_action)
                for node, prob in zip(nodes_for_action, non_zero_probabilities):
                    probabilities.append([action, node, prob])
        return probabilities

    @staticmethod
    def generate_random_node(w: int, h: int) -> Node:
        """
        Generisemo random cvor, sa odredjenim sansama, tako da je naravno Regularan najucestaliji:
        - RegularNode sa nagradom -1: 10/18 vrv (~55.56%)
        - RegularNode nagrade -10: 2/18 vrv (~11.11%)
        - TerminalNode: 2/18 vrv (~11.11%), nema nagrade
        - WallNode: 2/18 vrv (~11.11%)
        - TeleportNode: 2/18 vrv (~11.11%)
        """
        prob = rdm.randint(1, 18)
        if prob < 11:
            return RegularNode(-1, w, h)
        elif prob < 13:
            return RegularNode(-10, w, h)
        elif prob < 15:
            return TerminalNode(-1, w, h)
        elif prob < 17:
            return WallNode(w, h)
        else:
            return TeleportNode(-2, w, h)

    @staticmethod
    def generate_probabilities(cells: list) -> np.ndarray:
        probabilities = np.random.rand(len(cells))
        #normalizujemo
        probabilities /= sum(probabilities)
        #sada je ukupna suma 1
        return probabilities

    def print_graph(self):
        #prikazuje graf, u matricnom formatu, sa pozicijama, akcijama i vrednostima
        values_graph = {}
        for node in self.graph:
            values_graph[node.get_position()] = []
            for [action, next_node, prob] in self.graph[node]:
                values_graph[node.get_position()].append(
                    [action, next_node.get_position(), prob]
                )
        self.print_values(values_graph)
        return

    def print_values(self, g: dict):
        #ispisujemo vrednosti grafa
        # g je recnik gde je kljuc pozicija cvora, a vrednost je lista akcija, pozicija sledeceg cvog i verovatnoca
        print(
            "\n----------------------------------------------- MAZE GRAF "
            "------------------------------------------------ "
        )
        print(" ")
        for node in g:
            if self.get_current_pos_node(node).get_reward() == -10:
                print(node, "* : ", g[node])  # za regular bez kazne
                print(" ")
            else:
                print(node, ": ", g[node])
                print(" ")
        return

    @staticmethod
    def is_terminal(node: Node) -> bool:

        return node.is_terminal()

    def is_terminal_pos(self, position: tuple) -> bool:

        return self.get_current_pos_node(position).is_terminal()

    def get_graph(self) -> dict:

        return self.graph

    def get_current_pos_node(self, pos: tuple) -> Node:

        for node in self.graph:
            if node.get_position() == pos:
                return node
        raise Exception("Invalid position given.")

    def random_not_wall(self, nodes: list, depth: int = 0, max_depth: int = 20) -> Node:
        #vraca random cvor, koji nije Wall, a ako jeste rukurizvno se poziva ova funkcija opet, sve dok nije ili se ne dostigne maksimalna dubina rekurzije koja je predefinisana sa 20
        if depth > max_depth:
            raise RecursionError("Maximum depth recursion je dostignut u random_not_wall funkciji")

        random_node = rdm.choice(nodes)
        if isinstance(random_node, WallNode):
            return self.random_not_wall(nodes, depth + 1, max_depth)

        return random_node

    def get_action_probabilities(self, node: Node, action: int) -> list:

        #Vraca listu tupleova koji sadrze sledeci cvor i vrv da se dodje iz trenutnog cvora, za zadatu akciju

        action_probabilities = []
        for transition in self.graph[node]:
            transition_action, next_node, probability = transition
            if transition_action == action:
                action_probabilities.append((next_node, probability))

        return action_probabilities

    def get_next_node(self, nodes_probs: list) -> Node:
        #biramo cvor na osnovu liste verovatnoca
        if not nodes_probs or not any(prob for _, prob in nodes_probs):
            #vrati prvi cvor u listi kao default ili neki predefinisan cvor
            return nodes_probs[0][0] if nodes_probs else self.get_default_node()

        probabilities = [probability for _, probability in nodes_probs]
        # uzmi random indeks prema rasprostranjenosti verovatnoca, gde svaki indeks odgovara cvoru sa iz action_probabilities
        index = np.random.choice(len(probabilities), p=probabilities)
        return nodes_probs[index][0]

    def get_default_node(self):
        # vrati prvi cvor iz grafa
        return next(iter(self.get_graph()), None)

def get_node_color(cell: Node) -> str:
    if isinstance(cell, RegularNode) and cell.get_reward() == -10:
        return "red"
    elif isinstance(cell, RegularNode) and cell.get_reward() == -1:
        return "gray"
    elif isinstance(cell, WallNode):
        return "black"
    elif isinstance(cell, TerminalNode):
        return "blue"
    else:
        return "green"

def plot_maze_graph(env: MazeEnvironment):
    #ploltujemo okruzenje, ali Wall nema grane ni ka njemu ni iz njega
    g = nx.DiGraph()
    graph = env.get_graph()
    edge_labels = {}
    pos = {}  # mesto, recnik, gde belezimo pozicije
    offset = 0.1  # za podesavanje labela grana
    action_colors = {
        1: "magenta",
        2: "orange",
        3: "cyan"
    }

    teleport_edges = []  # lista sa granama iz teleporta
    normal_edges = []  # lista koja ima ostale grane

    for node in graph:
        position = node.get_position()
        pos[node] = position
        g.add_node(node, pos=position)

        if not isinstance(node, WallNode):
            for action, next_node, probability in graph[node]:
                if (action != 0 and probability != 0 and not isinstance(next_node, WallNode)):
                    edge_color = action_colors.get(action, "black")
                    if isinstance(node, TeleportNode):
                        teleport_edges.append((node, next_node, edge_color))
                    else:
                        normal_edges.append((node, next_node, edge_color))
                    edge_labels[(node, next_node)] = f"{next_node.get_reward()}"

    # crtanje covorova
    node_colors = {node: get_node_color(node) for node in g.nodes()}
    node_color_list = [node_colors[node] for node in g.nodes()]
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(g, pos, node_color=node_color_list, node_size=700)

    # crtanje normalnih grana
    for source, target, color in normal_edges:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[(source, target)],
            edge_color=color,
            arrowstyle="->",
            arrowsize=30,
            connectionstyle="arc3,rad=0.2",
        )

    # crtanje teleport grana, one su isprekidane
    for source, target, color in teleport_edges:
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[(source, target)],
            edge_color=color,
            style="dashed",
            alpha=0.5,
            arrowstyle="->",
            arrowsize=30,
            connectionstyle="arc3,rad=0.2",
        )

    node_labels = {node: f"{pos[node]}" for node in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_color="white")

    # podesevanje labela pozicije za labele grana
    edge_labels_pos = {
        edge: [
            (pos[edge[0]][0] + pos[edge[1]][0]) / 2 + offset,
            (pos[edge[0]][1] + pos[edge[1]][1]) / 2 + offset,
        ]
        for edge in edge_labels
    }

    for edge, new_pos in edge_labels_pos.items():
        label = edge_labels[edge]
        plt.text(new_pos[0], new_pos[1], label, fontsize=10, color="red")

    # sta je koji cvor, legenda kao
    node_type_colors = {
        "Regular Node": "gray",
        "Terminal Node": "blue",
        "Wall Node": "black",
        "Teleport Node": "green",
    }
    node_legends = [
        plt.Line2D([0], [0], color=color, marker="o", linestyle="", markersize=10)
        for color in node_type_colors.values()
    ]
    action_legends = [
        plt.Line2D([0], [0], color=color, marker="", linestyle="-")
        for color in action_colors.values()
    ]

    legend1 = plt.legend(
        node_legends, node_type_colors.keys(), title="Node Types", loc="upper left"
    )
    plt.gca().add_artist(legend1)
    plt.legend(
        action_legends,
        [f"Action {a}" for a in action_colors.keys()],
        title="Actions",
        loc="lower left",
    )

    plt.axis("off")
    plt.show()

def init_v_values(env: MazeEnvironment) -> dict:
    #incijalna vrednost netreminalnih cvorova je random negativna vrednost, dok je vrednost terminalnog cvora 0
    #kao izlaz imamo dict. sa kljucem kao tupple sa pozicijama, a vrednsti recnika su inicijalne v vrednosti

    return {node.get_position(): -20 * random() if not env.is_terminal(node) else 0 for node in env.get_graph()}


def init_q_values(env: MazeEnvironment) -> dict:
    q_values = {}
    for node in env.get_graph():
        for action in ACTIONS:
            if env.is_terminal(node):
                q_values[node.get_position(), action] = 0
            else:
                q_values[node.get_position(), action] = -20 * random()
    return q_values


def calculate_v_value(
    env: MazeEnvironment, position: tuple, values: dict, gamma: float
) -> float:
    # ulazni argumenti su pozicija cvora, trenutni recnik v vrednosti i gamma
    # izlaz je nova v vrednost za taj cvor, ako je zid u pitanju onda je fiksna velika negativna vrednost

    node_at_position = env.get_current_pos_node(position)
    action_values = []  # belezi izracunatu vrednost za svaku akciju

    for action in ACTIONS:
        value_for_action = 0
        for next_node, transition_probability in env.get_action_probabilities(
            node_at_position, action
        ):
            future_value = values[next_node.get_position()]
            reward = next_node.get_reward()
            value_for_action += transition_probability * (reward + gamma * future_value)

        action_values.append(value_for_action)

    # odabere najvecu vrednost od svih
    # ako je zid daj -100
    return max(action_values) if max(action_values) != 0 else -100

def calculate_q_value(
    env: MazeEnvironment, state: tuple, values: dict, gamma: float
) -> float:

    node_at_state = env.get_current_pos_node(state[0])
    action_value_contributions = []

    for next_node, transition_prob in env.get_action_probabilities(
        node_at_state, state[1]
    ):
        future_values = [
            values[next_node.get_position(), next_action] for next_action in ACTIONS
        ]
        best_future_value = max(future_values)
        reward = next_node.get_reward()

        value_contribution = transition_prob * (reward + gamma * best_future_value)
        action_value_contributions.append(value_contribution)
    return sum(action_value_contributions) if action_value_contributions else -100

def async_update_all_values(
    env: MazeEnvironment, values: dict, gamma: float, q_function: bool
) -> dict:
    for state_action_pair in values:
        if q_function:
            # kad update radimo, provera da nije u pitanju terminal
            if not env.is_terminal_pos(
                state_action_pair[0]
            ):
                values[state_action_pair] = calculate_q_value(
                    env, state_action_pair, values, gamma
                )
        else:
            # kad updateujemo v vrednosti, stanje je zapravo pozicija
            position = (
                state_action_pair  # ovde je to, pozicija je zapravo stanje
            )
            if not env.is_terminal_pos(position):
                values[position] = calculate_v_value(env, position, values, gamma)

    return values

def value_iteration(
    env: MazeEnvironment,
    gamma: float,
    convergence_threshold: float,
    max_iterations: int = 100,
    use_q_function: bool = False,
) -> tuple:

    current_values = init_q_values(env) if use_q_function else init_v_values(env)
    for iteration_number in range(max_iterations):
        previous_values = copy(current_values)
        updated_values = async_update_all_values(
            env, current_values, gamma, use_q_function
        )
        max_error = max(
            abs(updated_values[state] - previous_values[state])
            for state in current_values
        )

        if max_error < convergence_threshold:
            return updated_values, iteration_number + 1

        current_values = updated_values

    return current_values, iteration_number + 1

def best_action_min_arg(actions_probs: list) -> int:
    max_probability = max(prob for _, prob in actions_probs)
    max_probability_elements = [
        (action, prob) for action, prob in actions_probs if prob == max_probability
    ]

    min_action = min(action for action, _ in max_probability_elements)
    min_action_element = [
        (action, prob)
        for action, prob in max_probability_elements
        if action == min_action
    ][0][0]

    return min_action_element

def greedy_action(
    env: MazeEnvironment,
    current_node: Node,
    values: dict,
    gamma: float,
    use_q_values: bool = False,
) -> int:
    action_values = []
    #iteracija kroz akcije
    for action in ACTIONS:
      #reset koristi
        expected_utility = 0
        #idemo kroz sledece cvorove
        for next_node, transition_prob in env.get_action_probabilities(
            current_node, action
        ):
            if use_q_values:
                future_values = [
                    values[next_node.get_position(), next_action]
                    for next_action in ACTIONS
                ]
                best_future_value = max(future_values)
            else:
                best_future_value = values[next_node.get_position()]

            expected_utility += transition_prob * (
                next_node.get_reward() + gamma * best_future_value
            )

        action_values.append((action, expected_utility))

    # vraca tu najbolju akciju
    return best_action_min_arg(action_values) if action_values else None


def generate_optimal_policy(
    env: MazeEnvironment, values: dict, gamma: float, use_q_values: bool = False
) -> dict:
    optimal_policy = {}
    for node in env.get_graph():
        if not (node.is_terminal() or not node.is_steppable()):
            node_position = node.get_position()
            best_action = greedy_action(env, node, values, gamma, use_q_values)
            optimal_policy[node_position] = best_action

    return optimal_policy

def generate_random_policy(env: MazeEnvironment) -> dict:
    policy = {}
    for s in env.get_graph():
        policy[s.get_position()] = rdm.choice(ACTIONS)
    return policy

def evaluate_values(
    env: MazeEnvironment,
    policy: dict,
    gamma: float,
    convergence_threshold: float,
    use_q_values: bool = False,
) -> dict:
    current_values = init_q_values(env) if use_q_values else init_v_values(env)
    updated_values = copy(current_values)

    while True:
        for state in current_values:
            node = env.get_current_pos_node(state[0] if use_q_values else state)

            if isinstance(node, WallNode):
                updated_values[state] = -100
            elif isinstance(node, TerminalNode):
                updated_values[state] = 0
            else:
                chosen_action = policy[node.get_position()]
                transition_probs = env.get_action_probabilities(node, chosen_action)

                if use_q_values:
                    update_state_value_q(
                        state, updated_values, transition_probs, gamma, policy
                    )
                else:
                    update_state_value_v(state, updated_values, transition_probs, gamma)

        max_error = max(
            abs(updated_values[s] - current_values[s]) for s in current_values
        )
        if max_error < convergence_threshold:
            return updated_values

        current_values = updated_values
        
def update_state_value_q(state, values, transition_probs, gamma, policy):
    """    Args:
        state (tuple);
        values (dict): dict sa trenutnim Q vrednostima za svaki stanje - akcija par;
        transition_probs (list): lista taplova sledece_stanje, verovatnoca ;
        gamma (float);
        policy (dict): trenutna politika koju gledamo, ona zapravo mapira stanja akcijama
nema izlaza jer ovo samo azurira vec postojece q vrednosti
    """
    #prolaz kroz sledeca stanja i njihove verovatnoce
    for next_node, prob in transition_probs:
        if isinstance(next_node, TerminalNode):
            values[state] = next_node.get_reward()
            #sledeca akcija zavisi od politike
        else:
            next_action = policy[next_node.get_position()]
            #racunamo dobit
            values[state] = (
                next_node.get_reward()
                + gamma * values[next_node.get_position(), next_action]
            )

def update_state_value_v(state, values, transition_probs, gamma):

    value_sum = 0
    for next_node, prob in transition_probs:
        value_sum += prob * (
            next_node.get_reward() + gamma * values[next_node.get_position()]
        )
    values[state] = value_sum

def greedy_policy(env, values, gamma, q_function):
    return generate_optimal_policy(env, values, gamma, q_function)

def policy_iteration(
    env: MazeEnvironment,
    gamma: float,
    convergence_threshold: float,
    use_q_values: bool = False,
) -> dict:
    """
    Izlaz:
        dict: optimalna politika kao dict gde su kljucevi pozicije a vrednosti su akcije.
    """
    current_policy = generate_random_policy(env)
    while True:
        value_estimates = evaluate_values(
            env, current_policy, gamma, convergence_threshold, use_q_values
        )
        improved_policy = greedy_policy(env, value_estimates, gamma, use_q_values)

        if improved_policy == current_policy:
            return current_policy

        current_policy = improved_policy

#dimenzije grafa
dims = (3, 3)
#okruzenje
en = MazeEnvironment(dims)

#izlazi value)iterations su current_values i iteracija+1
v, v_it = value_iteration(en, 0.9, 0.01)
q, q_it = value_iteration(en, 0.9, 0.01, use_q_function=True)

# V tabela
v_table = PrettyTable()
v_table.field_names = ["Pozicija", "V vrednost"]
for position, value in v.items():
    v_table.add_row([position, value])

print(
    "\n----------------------------------- GOTOV ALGORITAM ITERIRANJA PO VREDNOSTIMA -----------------------------------\n"
)
print(f"Konacne V vrednosti po iteracijama za broj iteracija: {v_it}")
print(v_table)

# Q values table
q_table = PrettyTable()
q_table.field_names = ["Stanje-Akcija", "Q vrednost"]
for state_action, value in q.items():
    q_table.add_row([state_action, value])

print(f"\nKonacne Q vrednosti posle iteracija za broj iteracija: {q_it}")
print(q_table)

optimal_pol_v = generate_optimal_policy(en, v, 0.9)
optimal_pol_q = generate_optimal_policy(en, q, 0.9, use_q_values=True)

# Optimalna politika posle iteriranja V vrednosti
optimal_v_table = PrettyTable()
optimal_v_table.field_names = ["Pozicija", "Optimalna Akcija"]
for position, action in optimal_pol_v.items():
    optimal_v_table.add_row([position, action])

print(
    "\n---------------------------------- Optimalna Politika posle iteriranja po vrednostima ---------------------------------\n"
)
print("Optimalna politika posle iteriranja kroz V vrednosti je:")
print(optimal_v_table)

# Optimalna politika za Q tabelu
optimal_q_table = PrettyTable()
optimal_q_table.field_names = ["Pozicija", "Optimalna akcija"]
for position, action in optimal_pol_q.items():
    optimal_q_table.add_row([position, action])

print("\nOptimalna politika posle iteriranja kroz Q tabelu:")
print(optimal_q_table)

optimal_pol_pi_v = policy_iteration(en, 0.9, 0.01)
optimal_pol_pi_q = policy_iteration(en, 0.9, 0.01, use_q_values=True)

# Iteriranje politikama kroz V vrednosti
pi_v_table = PrettyTable()
pi_v_table.field_names = ["Pozicija", "Optimalna Akcija"]
for position, action in optimal_pol_pi_v.items():
    pi_v_table.add_row([position, action])

print(
    "\n---------------------------------- Optimalna akcija po iteriranju kroz V vrednosti --------------------------------\n"
)
print("Optimalna politika posle iteriranja politikama za V je :")
print(pi_v_table)

# Iteriranje politikama Q
pi_q_table = PrettyTable()
pi_q_table.field_names = ["Pozicija", "Optimalna Akcija"]
for position, action in optimal_pol_pi_q.items():
    pi_q_table.add_row([position, action])

print("\Optimalna politika posle iteracije politikama koriscenjem Q:")
print(pi_q_table)

print("\n Boje akcija:  \n1 - roze, \n2 - narandzasta, \n3 -  plava\n")

print("\n\n Boje Cvorova:  \nKazneno - crveno, \nRegularno - sivo, \nWall -  crno, \nTerminal plavo \nNeki drugi cvor - zeleno \n")
en.print_graph()
plot_maze_graph(en)
