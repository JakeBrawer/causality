import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from causalgraphicalmodels.csm import StructuralCausalModel, linear_model
from .sgd import SGD

DATA_FILE = "./training_data.csv"

def rand_n_ints(low=0, high=10, size=(10)):
    return 1. * np.random.random_integers(low, high, size=size)

def normalize(v, vec):
    vec = np.array(vec)
    norm=np.linalg.norm(vec, ord=1)
    print(norm)
    if norm==0:
        norm=np.finfo(vec.dtype).eps
    return v/norm

def do_multiple(variable_list, scm):
    scm_dos = scm.do(variable_list.pop())
    for v in variable_list:
        scm_dos = scm_dos.do(v)

    return scm_dos

def simulate_pushes(scm, num_pushes):
    ds = scm.sample(n_samples=1)
    plt.plot(ds.goal_x, ds.goal_y, 'ro')

    for i in range(0,num_pushes):
        intervention_vars = {"init_x": ds.final_x, "init_y":  ds.final_y,
                             "goal_x":  ds.goal_x, "goal_y": ds.goal_y}

        scm_do            = do_multiple(list(intervention_vars), scm)

        print(ds.head())

        plt.plot(ds.init_x, ds.init_y, 'bo')
        plt.text(ds.init_x * (1 + 0.01), ds.init_y * (1 + 0.01) , i, fontsize=12)
        # plt.plot(ds.final_x, ds.final_y, 'go')
        plt.quiver(ds.init_x,ds.init_y, ds.dist_x, ds.dist_y,)

        ds = scm_do.sample(n_samples=1, set_values=intervention_vars)

    plt.show()


def get_coefficients(df, child, parents):
    y = df[child].values
    X = np.column_stack((df[i].values for i in parents))

    reg = LinearRegression().fit(X, y)

    coefs     = reg.coef_
    intercept = reg.intercept_

    print("Score: {}".format(reg.score(X, y)))

    return intercept, coefs


def scm_to_linear_scm(scm, ds, n_samples=100):
    linear_scm_model = {}

    for node, model in scm.assignment.items():
        print("Node: {}  Items: {}".format(node, model))
        parents = model.parents
        if parents:
            intercept, coefs       = get_coefficients(ds, node, parents)
            linear_scm_model[node] = linear_model(parents, coefs, offset=intercept, noise_scale=.1)
            # print("mu: {}, std: {}".format(mu, std))
            print("PARENTS: {}".format(parents))
            print("COEFFS: {}".format(coefs))

        else:
            mu  = np.mean(ds[node])
            std = np.std(ds[node])

            structural_eq          = lambda n_samples: np.random.normal(loc=mu, scale=std, size=n_samples)
            linear_scm_model[node] = structural_eq

    return StructuralCausalModel(linear_scm_model)

def compare_scms(scm1, scm2):
    intervention_vars = {"init_x": np.arange(0,3), "init_y": np.arange(0,3),
                         "goal_x": np.arange(7,10), "goal_y": np.arange(7,10)}


    scm1_do = do_multiple(list(intervention_vars), scm1)
    scm2_do = do_multiple(list(intervention_vars), scm2)

    print("SCM 1")
    ds1 = scm1_do.sample(n_samples=3, set_values=intervention_vars)
    print(ds1.head())
    print("SCM 2")
    ds2 = scm2_do.sample(n_samples=3, set_values=intervention_vars)
    print(ds2.head())
# scm = StructuralCausalModel({
#     "goal_x": lambda  n_samples: np.random.normal(loc = 2., scale=2.0, size=n_samples),
#     "goal_y": lambda  n_samples: np.random.normal(loc = 2., scale=2.0, size=n_samples),
#     "init_x": lambda  n_samples: np.random.normal(loc = 8., scale=2.0, size=n_samples),
#     "init_y": lambda  n_samples: np.random.normal(loc = 8., scale=2.0, size=n_samples),
#     "dist_x": lambda goal_x, goal_y, init_x, init_y, n_samples: np.random.normal(
#         loc=normalize(goal_x - init_x, [goal_x - init_x, goal_y - init_y]), scale=0.2),
#     "dist_y": lambda goal_x, goal_y, init_x, init_y, n_samples: np.random.normal(
#         loc=normalize(goal_y - init_y, [goal_x - init_x, goal_y - init_y]), scale=0.2),
#     "final_x":lambda dist_x, init_x, n_samples: np.random.normal(init_x + dist_x, scale=0.1),
#     "final_y":lambda dist_y, init_y, n_samples: np.random.normal(init_y + dist_y, scale=0.1),
# })

def online_learning_example(it):

    # Initialize weights to random value
    weight_final_x    = np.random.randint(10, size =2)
    weight_final_y    = np.random.randint(10, size =2)

    ground_truth_scm = StructuralCausalModel({
        "init_x": lambda  n_samples: np.random.normal(loc = 8., scale=2.0, size=n_samples),
        "init_y": lambda  n_samples: np.random.normal(loc = 8., scale=2.0, size=n_samples),
        "push_x": lambda  n_samples: np.random.normal(loc = 0., scale=1.0, size=n_samples),
        "push_y": lambda  n_samples: np.random.normal(loc = 0., scale=1.0, size=n_samples),
        "final_x":linear_model(["init_x", "dist_x"], [1, .1], noise_scale=.01) ,
        "final_y":linear_model(["init_y", "dist_y"], [1, .1], noise_scale=.01) ,
    })

    learned_scm = StructuralCausalModel({
        "init_x": lambda  n_samples: np.random.normal(loc = 8., scale=2.0, size=n_samples),
        "init_y": lambda  n_samples: np.random.normal(loc = 8., scale=2.0, size=n_samples),
        "push_x": lambda  n_samples: np.random.normal(loc = 0., scale=1.0, size=n_samples),
        "push_y": lambda  n_samples: np.random.normal(loc = 0., scale=1.0, size=n_samples),
        "final_x":linear_model(["init_x", "dist_x"], weight_final_x, noise_scale=.01) ,
        "final_y":linear_model(["init_y", "dist_y"], weight_final_y, noise_scale=.01) ,
    })

    df_gt                = ground_truth_scm.sample(n_samples =it)
    intervention_vars = {"init_x": df_gt.final_x, "init_y":  df_gt.final_y,
                         "push_x":  df_gt.push_x, "push_y": df_gt.push_y}
    learned_scm_do    = do_multiple(list(intervention_vars), learned_scm)



scm = StructuralCausalModel({
    "goal_x": lambda  n_samples: np.random.normal(loc = 2., scale=2.0, size=n_samples),
    "goal_y": lambda  n_samples: np.random.normal(loc = 2., scale=2.0, size=n_samples),
    "init_x": lambda  n_samples: np.random.normal(loc = 8., scale=2.0, size=n_samples),
    "init_y": lambda  n_samples: np.random.normal(loc = 8., scale=2.0, size=n_samples),
    "dist_x": linear_model(["goal_x", "init_x"], [1, -1], noise_scale=.1),
    "dist_y": linear_model(["goal_y", "init_y"], [1, -1], noise_scale=.1),
    "final_x":linear_model(["init_x", "dist_x"], [1, .2], noise_scale=.1) ,
    "final_y":linear_model(["init_y", "dist_y"], [1, .2], noise_scale=.1) ,
})

ds = scm.sample(n_samples=1000)

# ds.to_csv(DATA_FILE)
# print(list(scm.assignment.items()))
# print(ds["init_x"].values)
# print(get_coefficients(ds,"dist_x", ["goal_x", "goal_y", "init_x", "init_y"]))

lin_scm = scm_to_linear_scm(scm, ds)

simulate_pushes(lin_scm, 10)

# compare_scms(scm, lin_scm)

