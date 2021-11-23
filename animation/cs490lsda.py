from manim import *
import random

class BTree(Scene):
    def construct(self):
        root = Dot([-2, 0, 0])
        layer1 = [Dot([2, x, 0]) for x in range(-2, 3)]
        branches1= [Line(root.get_center(), node.get_center()) for node in layer1]

        t1 = Tex(r"\justifying{B-Trees are a rooted graph data structure that generalize binary trees to have multiple children instead of just two.}", font_size=24)
        t1.shift(UP * 3)
        self.play(Write(t1))
        self.play(Create(root))

        for node, line in zip(layer1, branches1):
            self.play(AnimationGroup(Create(node), Create(line), lag_ratio=0.5))
        self.play(FadeOut(t1))

        t2 = Tex(r"\justifying{Each node, then, is associated with a collection of elements, instead of a single value.}", font_size=24)
        t2.shift(UP * 3)
        self.play(Write(t2))

        data = sorted([random.randrange(0, 1000) for _ in range(25)])
        nodedata = [data[5 * i: 5 * i + 5] for i in range(5)]
        tables = [IntegerTable([d], include_outer_lines=True, h_buff=1.0) for d in nodedata]
        for table, node in zip(tables, layer1):
            table.scale(0.5)
            table.next_to(node)
        self.play(*[Create(table) for table in tables])
        
        self.play(FadeOut(t2))
        rootdata = [[d[0]] for d in nodedata]
        roottable = IntegerTable(rootdata, include_outer_lines=True)
        roottable.scale(0.5)
        roottable.next_to(root, LEFT)
        self.play(Create(roottable))

        t3 = Tex(r"\justifying{Let us consider the behavior of a specific node when performing search.}", font_size=24)
        t3.shift(UP * 3)
        self.play(Write(t3))

        self.play(Indicate(root), Indicate(roottable))
        self.play(FadeOut(t3))

        t4 = Tex(r"\justifying{Just like with binary trees, for a given query value $x$, we match the relative value of $x$ against each value.}", font_size=24)
        t4.shift(UP * 3)
        self.play(Write(t4))
        self.wait(1)
        self.play(FadeOut(t4))

        for i, num in enumerate(rootdata):
            if i == 0:
                prev = num
                continue
            else:
                ti = Tex(fr"\justifying{{If ${prev[0]} \leq x < {num[0]}$, then if $x$ exists, it must be in this subtree.}}", font_size=24)
            ti.shift(UP * 3)
            cell = roottable.get_cell((i, 1), color=RED)
            self.play(Write(ti), FadeIn(cell))
            self.play(Indicate(branches1[i - 1], scale_factor=1), Indicate(layer1[i - 1]), Indicate(tables[i - 1]))
            self.play(FadeOut(ti), FadeOut(cell))
            prev = num
        tn = Tex(fr"\justifying{{If $x \geq {num[0]}$, then if $x$ exists, it must be in this subtree.}}", font_size=24)
        tn.shift(UP * 3)
        cell = roottable.get_cell((5, 1), color=RED)
        self.play(Write(tn), FadeIn(cell))
        self.play(Indicate(branches1[4], scale_factor=1), Indicate(layer1[4]), Indicate(tables[4]))
        self.play(FadeOut(tn), FadeOut(cell))

        t5 = Tex(r"\justifying{Each node, then, is really a node of a decision tree. We consider the value of an attribute, and decide which child to proceed onto. What if instead of selecting the child based on its relative range, we use some other ML inspired techiques?}", font_size=24)
        t5.shift(UP * 3)
        self.play(Write(t5))
        self.play(FadeOut(t5))

        t6 = Tex(r"Let's focus on just the root node.", font_size=24)
        t6.shift(UP * 3)
        self.play(Write(t6))
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob != root and mob != roottable])

        t7 = Tex(r"\justifying{If we enumerate each child from $1$ to $n$, then we can capture this categorizing procedure as a piecewise function.}", font_size=24)
        t7.shift(UP * 3)
        self.play(Write(t7))

        func = MathTex(
            fr"f(x) = \begin{{cases}} 1 & {rootdata[0][0]} \leq x < {rootdata[1][0]} \\ 2 & {rootdata[1][0]} \leq x < {rootdata[2][0]} \\ 3 & {rootdata[2][0]} \leq x < {rootdata[3][0]} \\ 4 & {rootdata[3][0]} \leq x < {rootdata[4][0]} \\ 5 & x \geq {rootdata[4][0]} \end{{cases}}"
        )
        self.play(FadeOut(t7))
        self.play(Transform(VGroup(root, roottable), func))

        t8 = Tex(r"\justifying{This leads to the following graph.}", font_size=24)
        self.play(Write(t8))

        ax = Axes(
            x_range=[0, 1000, 100],
            y_range=[-1, 5, 1],
            tips=False,
            axis_config={"include_numbers": True}
        )

        def inner(x):
            nonlocal nodedata
            for i, y in enumerate(nodedata):
                if x <= y:
                    return i - 1
            return 4
        
        graph = ax.get_graph(inner, x_range=[0, 1000], use_smoothing=False)

        self.play(Create(ax))
        self.play(Create(graph))

            


        


