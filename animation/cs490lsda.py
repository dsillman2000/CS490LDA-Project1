from manim import *
import random

class BTree(MovingCameraScene):
    def construct(self):
        root = Dot([-2, 0, 0])
        layer1 = [Dot([2, x, 0]) for x in range(-2, 3)]
        branches1= [Line(root.get_center(), node.get_center()) for node in layer1]

        t1 = Tex(r"\justifying{B-Trees are a rooted graph data structure that generalize binary trees to have multiple children instead of just two.}")
        t1.shift(UP * 3)
        t1.scale(0.5)
        self.play(Write(t1))
        self.play(Create(root))

        for node, line in zip(layer1, branches1):
            self.play(AnimationGroup(Create(node), Create(line), lag_ratio=0.5))
        self.play(FadeOut(t1))

        t2 = Tex(r"\justifying{Each node, then, is associated with a collection of elements, instead of a single value.}")
        t2.shift(UP * 3)
        t2.scale(0.5)
        self.play(Write(t2))

        data = sorted([random.randrange(0, 1000) for _ in range(25)])
        nodedata = [data[5 * i: 5 * i + 5] for i in range(5)]
        tables = [IntegerTable([d], include_outer_lines=True, h_buff=1.0) for d in nodedata]
        for table, node in zip(tables, layer1):
            table.scale(0.5)
            table.next_to(node)
        self.play(*[Create(table) for table in tables])
        
        self.play(FadeOut(t2))
        roottable = IntegerTable([[d[0]] for d in nodedata], include_outer_lines=True)
        roottable.scale(0.5)
        roottable.next_to(root, LEFT)
        self.play(Create(roottable))

        t3 = Tex(r"\justifying{Let us consider the behavior of a specific node when performing search.}")
        t3.shift(UP * 3)
        t3.scale(0.5)
        self.play(Write(t3))

        vg = VGroup(root, roottable)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(vg).set(height=vg.height))

