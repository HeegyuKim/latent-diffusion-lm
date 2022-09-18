from coop.models import Optimus

checkout_step = 2000

for i in range(0, 10000, 100):
    klw = Optimus.klw(i, checkout_step)
    print(i, klw)