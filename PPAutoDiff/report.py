import contextlib
from collections import namedtuple
from .actions import get_action

class Counter: 
    def __init__(self):
        self.clear()

    def clear(self):
        self.id = 0

    def get_id(self):
        ret = self.id
        self.id += 1
        return ret

ReportItem = namedtuple(
    "ReportItem",
    ["step", "input", "output", "net"],
)

class Report:
    def __init__(self, name):
        self.name = name
        self.items = []

    def put_item(self, inp, outp, net): 
        step = global_counter.get_id()
        self.items.append(ReportItem(
            step=step, 
            input=inp, 
            output=outp,
            net=net,
        ))

    def get_items(self):
        sorted(self.items, key=lambda x: x.step)
        return self.items
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        sorted(self.items, key=lambda x: x.step)
        strings = []
        strings.append("Report name is: " + self.name)
        for item in self.items:
            strings.append("    " + str(item.step) + ": [{}]".format(type(item.net)))
        return "\n".join(strings)

global_report = None
global_counter = Counter()

@contextlib.contextmanager
def report_guard(report):
    global global_report
    old = global_report
    try:
        global_report = report
        global_counter.clear()
        yield
    finally:
        global_report = old

def current_report():
    if global_report is None: 
        raise RuntimeError("Please call `current_report()` within contextmanager `report_guard(Report())`.")
    return global_report 

def print_report(torch_rep, paddle_rep, cfg):
    """
    TODO(@xiongkun): 
    More abundant printing methods can be supported later，For example, interactive printing mode，Tree Printing mode，Currently, only list printing is supported.
    """
    torch_items = torch_rep.get_items()
    paddle_items = paddle_rep.get_items()
    assert len(torch_items) == len(paddle_items), "Difference length of torch_items and paddel_items, make sure the paddle layer and torch module have the same valid sublayer."
    for torch_item, paddle_item in zip(torch_items, paddle_items):
        assert torch_item.step == paddle_item.step, "Different step id, please call rep.get_items instead of rep.items."
        act = get_action(torch_item.net, paddle_item.net)
        try: 
            act(torch_item, paddle_item, cfg)
        except Exception as e: 
            print ("FAILED !!!")
            print ("diff found in step: {}".format(torch_item.step))
            print ("type of layer is  : {} vs {}".format(type(torch_item.net), type(paddle_item.net)))
            print (str(e))
            return False
    print ("SUCCESS !!!")
    return True
    
