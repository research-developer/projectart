from projectart.reactive.config import ObjectTemplate
from projectart.reactive.objects import ReactiveObject


def test_from_template_copies_fields():
    t = ObjectTemplate(
        kind="ball", radius=0.04, shape="circle", mass=2.0, color="#f00", kinematic=False
    )
    o = ReactiveObject.from_template(7, t, x=0.5, y=0.5)
    assert (o.id, o.kind, o.radius, o.mass, o.shape, o.color) == (
        7, "ball", 0.04, 2.0, "circle", "#f00"
    )
    assert (o.x, o.y, o.vx, o.vy) == (0.5, 0.5, 0.0, 0.0)
    assert o.kinematic is False


def test_inv_mass_kinematic_is_zero():
    t = ObjectTemplate(kind="box", mass=1.0, kinematic=True)
    o = ReactiveObject.from_template(1, t, x=0, y=0)
    assert o.inv_mass == 0.0
    t2 = ObjectTemplate(kind="ball", mass=2.0, kinematic=False)
    o2 = ReactiveObject.from_template(2, t2, x=0, y=0)
    assert o2.inv_mass == 0.5
