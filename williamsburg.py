import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pickle
import plotly.express as px
EPSILON = 1e-3

stations = [
    (40.69978508399301, -73.95003262834037, 'Flushing Av'),
    (40.70391698978126, -73.94732896174611, 'Lorimer St'),
    (40.695327543912875, -73.94917432148505, 'Myrtle-Willoughby Avs'),
    (40.71178967550891, -73.94041959156081, 'Grand St'),
    (40.70765825814149, -73.93994752279038, 'Montrose Av'),
    (40.700273117710545, -73.94114915238782, 'Flushing Av'),
    (40.69711710313728, -73.93557015782827, 'Myrtle Av')
]


load_g = True
if not load_g:
    gdf = gpd.read_file('./data/tl_2022_us_zcta520/tl_2022_us_zcta520.shp')
    gdf = gdf[gdf['ZCTA5CE20'] == '11206']
    shape = gdf.iloc[0].geometry
    g = ox.graph_from_polygon(shape, network_type='drive', simplify=True)
    g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    with open('./data/williamsburg.pkl', 'wb') as file:
        pickle.dump(g, file)
else:
    with open('./data/williamsburg.pkl', 'rb') as file:
        g = pickle.load(file)
amts = [ox.nearest_nodes(g, lon, lat) for lat, lon, _ in stations]

debug = False
if debug:
    g = nx.ego_graph(g, amts[0], radius=1000, distance='length')
    g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    amts = [ox.nearest_nodes(g, lon, lat) for lat, lon, _ in stations]


class City:

    def __init__(self, g, rho=2):

        self.rho = rho

        self.g = g
        for _, data in self.g.nodes(data=True):
            data['amt'] = False
            data['inh'] = set()
            data['dow_thr'] = 0
            data['upk'] = None
            data['cmt'] = None
            data['pop_hist'] = []
            data['cmt_hist'] = []

        self.diam = nx.diameter(self.g, weight='length')
        self.dist = dict(nx.all_pairs_dijkstra_path_length(self.g, weight='length'))
        self.dist = pd.DataFrame.from_dict(self.dist).sort_index() / self.diam
        self.dist = self.dist.to_numpy()

        self.amts = None
        self.amts_dist = None

        self.agts = None
        self.agt_dows = None

    def set_amts(self, amts):
        self.amts = amts
        for u in self.amts:
            data = self.g.nodes[u]
            data['amt'] = True
            data['inh'] = None
            data['dow_thr'] = None
            data['upk'] = None
            data['cmt'] = None
            data['pop_hist'] = None
            data['cmt_hist'] = None
        self.amts_dist = np.array([min(self.dist[u][v] for v in self.amts) for u in self.g.nodes()])

    def set_agts(self, agts):
        self.agts = agts
        self.agt_dows = np.array([a.dow for a in self.agts])

    def update(self):
        for u, data in self.g.nodes(data=True):
            if data['amt']:
                continue
            pop = len(data['inh'])
            cmt = np.average(self.agt_dows, weights=[(1 - self.dist[u][a.u]) ** 2 for a in self.agts])
            if pop > 0:
                if pop < self.rho:
                    data['dow_thr'] = 0
                else:
                    data['dow_thr'] = sorted([a.dow for a in self.g.nodes[u]['inh']])[-self.rho]
                data['upk'] = True
            else:
                data['dow_thr'] = 0
                data['upk'] = False
            data['cmt'] = cmt

            data['pop_hist'].append(pop)
            data['cmt_hist'].append(cmt)

    def plot(self, cmap='YlOrRd', figkey=None):

        for u, data in self.g.nodes(data=True):
            if not data['amt']:
                data['dow'] = np.average(self.agt_dows, weights=[a.avg_probabilities[u] for a in self.agts])
                data['dow'] = (data['dow'] - min(city.agt_dows)) / (max(city.agt_dows) - min(city.agt_dows))
                data['pop'] = np.sum([a.avg_probabilities[u] for a in self.agts])
            else:
                data['dow'] = np.nan
                data['pop'] = np.nan

        no_agts = len(self.agts)
        node_size = [no_agts / 10 * data['pop'] if not data['amt'] else no_agts / 2.5 for _, data in self.g.nodes(data=True)]
        node_color = ox.plot.get_node_colors_by_attr(self.g, 'dow', start=0, stop=1, na_color='b', cmap=cmap)
        fig, ax = plt.subplots(figsize=(9, 6))
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap=plt.colormaps[cmap]), ax=ax, location='bottom', shrink=0.5, pad=0.05
        )
        cb.set_label('Expected Endowment', fontsize=14)
        ox.plot_graph(self.g, ax=ax, bgcolor='w', node_color=node_color, node_size=node_size)
        plt.show()
        if figkey is not None:
            plt.savefig('./figures/{0}.pdf'.format(figkey), bbox_inches='tight', format='pdf')

        # df_data = []
        # for a in self.agts:
        #     df_data.append(
        #         ((self.agt_dows < a.dow).mean(), np.sum([a.avg_probabilities[u] * self.amts_dist[u] for u in self.g.nodes()]), self.rho, a.alpha)
        #     )
        # df = pd.DataFrame(df_data, columns=['dow', 'exp_dist', 'rho', 'alpha'])
        # fig = px.scatter(
        #     df, x="dow", y="exp_dist",
        #     labels={'dow': 'Endowment Quantile', 'exp_dist': 'Expected Distance to Amenity'}
        # )
        # fig.update_yaxes(range=[0, max(self.amts_dist[u] for u in self.g.nodes())])
        # fig.show()
        #
        # df_data = []
        # for u, data in self.g.nodes(data=True):
        #     df_data.append((self.amts_dist[u], data['pop'], self.rho))
        # df = pd.DataFrame(df_data, columns=['dist', 'exp_pop', 'rho'])
        # fig = px.scatter(
        #     df, x="dist", y="exp_pop",
        #     labels={'dist': 'Distance to Amenity', 'exp_pop': 'Expected Population'}
        # )
        # fig.update_yaxes(range=[0, 8])
        # fig.show()


class Agent:

    def __init__(self, i, dow, city, alpha=0.5):

        self.i = i
        self.dow = dow
        self.city = city
        self.alpha = alpha

        self.weights = None
        self.probabilities = None
        self.tot_probabilities = None
        self.avg_probabilities = None
        self.u = None

        self.reset()

    def __hash__(self):
        return hash(self.i)

    def __eq__(self, other):
        return self.i == other.i

    def reset(self):
        self.weights = np.array([1.0 if not data['amt'] else 0 for _, data in city.g.nodes(data=True)])
        self.probabilities = np.array(self.weights / self.weights.sum())
        self.tot_probabilities = self.probabilities.copy()
        self.u = np.random.choice(self.city.g.nodes(), p=self.probabilities)
        self.city.g.nodes[self.u]['inh'].add(self)

    def act(self):
        self.city.g.nodes[self.u]['inh'].remove(self)
        self.u = np.random.choice(self.city.g.nodes(), p=self.probabilities)
        self.city.g.nodes[self.u]['inh'].add(self)

    def learn(self):
        for u in self.city.g.nodes():
            if not self.city.g.nodes[u]['amt']:
                self.weights[u] *= (1 - EPSILON * self.cost(u))
        self.probabilities = np.array(self.weights / self.weights.sum())
        self.tot_probabilities += self.probabilities

    def cost(self, u):
        aff = int(self.dow >= self.city.g.nodes[u]['dow_thr'])
        loc = np.exp(- (1 - self.alpha) * self.city.amts_dist[u])
        upk = int(self.city.g.nodes[u]['upk'])
        cmt = np.exp(- self.alpha * np.abs(self.dow - self.city.g.nodes[u]['cmt']))
        c = 1 - aff * loc * upk * cmt
        return c


rho_l = [4, 8]
alpha_l = [0.75]
t_max_l = [5000, 10000, 15000, 20000]
tau = 0.5
run_experiments = True
plot_cities = True
cty_key = 'williamsburg'
# cty_key = 'city'

n = g.number_of_nodes() - len(amts)
if run_experiments:
    for rho in rho_l:
        for alpha in alpha_l:

            np.random.seed(0)

            city = City(g, rho=rho)
            city.set_amts(amts)

            agt_dows = np.diff([1 - (1 - x) ** tau for x in np.linspace(0, 1, n + 1)])
            agts = [Agent(i, dow, city, alpha=alpha) for i, dow in enumerate(agt_dows)]

            city.set_agts(agts)
            city.update()

            for t in range(max(t_max_l)):
                print('t: {0}'.format(t))
                for a in agts:
                    a.act()
                city.update()
                for a in agts:
                    a.learn()

                if t + 1 in t_max_l:

                    for a in city.agts:
                        a.avg_probabilities = a.tot_probabilities / (t + 1)

                    with open('./data/{0}_{1}_{2}_{3}.pkl'.format(cty_key, rho, alpha, t + 1), 'wb') as file:
                        pickle.dump(city, file)

if plot_cities:
    for rho in rho_l:
        for alpha in alpha_l:
            for t_max in t_max_l:
                with open('./data/{0}_{1}_{2}_{3}.pkl'.format(cty_key, rho, alpha, t_max), 'rb') as file:
                    city = pickle.load(file)
                cmap = 'YlOrRd'
                figkey = './{0}_{1}_{2}_{3}'.format(cty_key, rho, alpha, t_max)
                city.plot(cmap=cmap, figkey=figkey)


# df_data = []
# for rho in rho_l:
#     for alpha in alpha_l:
#         with open('./data/{0}_{1}_{2}.pkl'.format(cty_key, rho, alpha), 'rb') as file:
#             city = pickle.load(file)
#         for u, data in city.g.nodes(data=True):
#             if not data['amt']:
#                 data['dow'] = np.average(city.agt_dows, weights=[a.probabilities[u] for a in city.agts])
#                 data['pop'] = np.sum([a.probabilities[u] for a in city.agts])
#             else:
#                 data['dow'] = np.nan
#                 data['pop'] = np.nan
#         for a in city.agts:
#             df_data.append(
#                 (a.dow, (city.agt_dows < a.dow).mean(), np.sum([a.probabilities[u] * city.amts_dist[u] for u in city.g.nodes()]), rho, alpha)
#             )
# df = pd.DataFrame(df_data, columns=['dow', 'dow_qtl', 'exp_dist', 'rho', 'alpha'])
# df['rho'] = df['rho'].astype(str)
# df['alpha'] = df['alpha'].astype(str)
# fig = px.scatter(
#     df, x='dow_qtl', y='exp_dist', color='rho', facet_col='alpha', color_discrete_sequence=px.colors.qualitative.T10,
#     labels={
#         'dow_qtl': 'Endowment Quantile',
#         'exp_dist': 'Expected Distance to Amenity',
#         'alpha': '$\lambda$',
#         'rho': r'$\rho$'
#     }
# )
# fig.update_yaxes(range=[0, 0.15])
# fig.show()
#
# df_data = []
# for rho in rho_l:
#     for alpha in alpha_l:
#         with open('./data/{0}_{1}_{2}.pkl'.format(cty_key, rho, alpha), 'rb') as file:
#             city = pickle.load(file)
#         for u, data in city.g.nodes(data=True):
#             if not data['amt']:
#                 data['dow'] = np.average(city.agt_dows, weights=[a.probabilities[u] for a in city.agts])
#                 data['pop'] = np.sum([a.probabilities[u] for a in city.agts])
#             else:
#                 data['dow'] = np.nan
#                 data['pop'] = np.nan
#         for u, data in city.g.nodes(data=True):
#             df_data.append((city.amts_dist[u], data['dow'], data['pop'], rho, alpha))
# df = pd.DataFrame(df_data, columns=['dist', 'exp_dow', 'exp_pop', 'rho', 'alpha'])
# df.dropna(inplace=True)
# df['rho'] = df['rho'].astype(str)
# df['alpha'] = df['alpha'].astype(str)
# fig = px.scatter(
#     df, x='dist', y='exp_pop', facet_col='alpha', facet_row='rho', color_discrete_sequence=px.colors.qualitative.T10,
#     labels={
#         'dist': 'Distance to Amenity',
#         'exp_dow': 'Expected Endowment',
#         'exp_pop': 'Expected Population',
#         'alpha': '$\lambda$',
#         'rho': r'$\rho$'
#     }
# )
# fig.show()
# fig = px.scatter(
#     df, x='dist', y='exp_dow', facet_col='alpha', facet_row='rho', color_discrete_sequence=px.colors.qualitative.T10,
#     labels={
#         'dist': 'Distance to Amenity',
#         'exp_dow': 'Expected Endowment',
#         'exp_pop': 'Expected Population',
#         'alpha': '$\lambda$',
#         'rho': r'$\rho$'
#     }
# )
# fig.show()
# fig = px.scatter(
#     df, x='dist', y='exp_dow', size='exp_pop', color='rho', facet_col='alpha', color_discrete_sequence=px.colors.qualitative.T10,
#     trendline="ols", trendline_options=dict(log_y=True), opacity=0.75,
#     labels={
#         'dist': 'Distance to Amenity',
#         'exp_dow': 'Expected Endowment',
#         'exp_pop': 'Expected Population',
#         'alpha': '$\lambda$',
#         'rho': r'$\rho$'
#     }
# )
# fig.update_traces(marker_size=3.75)
# fig.show()