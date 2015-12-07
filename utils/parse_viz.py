from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
import nltk
from graphviz import Graph

class ParseTreeViz:
    
    def __init__(self):
        pass
    
    @staticmethod
    def _isLeave(node):
        return not isinstance(node, Tree)
    
    #dfs traverse the tree
    @staticmethod
    def _traverse(tree):
        nodeMap = {}
        nodeMap[id(tree)] = ('node0', tree.label(), True if ParseTreeViz._isLeave(tree) else False)
        if ParseTreeViz._isLeave(tree):
            return nodeMap
        def _dfs(root, idx):
            for node in root:
                if ParseTreeViz._isLeave(node):
                    if isinstance(node, str):
                        rep = node
                    elif isinstance(node, tuple) or isinstance(node, list):
                        rep = r'/'.join(node)
                    elif hasattr(node, '__str__'):
                        rep = str(node)
                    else:
                        raise RuntimeError('the leave node can not be stringify.')
                    nodeMap[str(node)+str(id(root))] = ('node%d' % (idx,), rep , True)
                    idx += 1
                else:
                    nodeMap[id(node)] = ('node%d' % (idx,), node.label(), False)
                    idx  = _dfs(node, idx+1)
            return idx
        _dfs(tree, 1)
        return nodeMap
    
    @staticmethod
    def _buildGraph(tree, nodeMap, dot):
        
        ##add the nodes to the Graph
        for name, mark, isleave in sorted(nodeMap.values(), key=lambda x: int(x[0][4:])):
            if isleave:
                dot.node(name, mark, fontcolor='blue', fontname='Times-Bold')
            else:
                dot.node(name, mark)
        
        def _dfs(root):
            if (ParseTreeViz._isLeave(root)):
                return
            root_name = nodeMap[id(root)][0] if nodeMap.get(id(root)) else None
            if not root_name:
                return
            for node in reversed(root):
                if ParseTreeViz._isLeave(node):
                    dot.edge(root_name, nodeMap[str(node)+str(id(root))][0])
                else:
                    dot.edge(root_name, nodeMap[id(node)][0])
                    _dfs(node)
        _dfs(tree)
    
    @staticmethod
    def buildVizGraph(tree, name_, comment_=None):
        nmap = ParseTreeViz._traverse(tree)
        dot = Graph(name=name_, comment=comment_, node_attr={'shape': 'none'}, graph_attr={'nodesep': '0.1', 'ranksep': '0.2'})
        ParseTreeViz._buildGraph(tree, nmap, dot)
        return dot


class SyntaxTreeParser:
    def __init__(self):
        self.parser = StanfordParser()
        if not self.parser:
            raise RuntimeError('Stanford Parsre could not be initialized.')
    
    def raw_parse(self, sent):
        tree = next(self.parser.raw_parse(sent))
        return tree

    def parse(self, sent):
        one_sent = sent
        if len(sent[0]) == 1:
            one_sent = nltk.pos_tag(sent)
        tree = self.parser.tagged_parse(one_sent)
        return tree

from matplotlib import pyplot as plt
if __name__ == '__main__':
    sent = 'Considered the greatest 20th century novel written in English, in this edition Walter Gabler uncovers previously unseen text.'
    parser = SyntaxTreeParser()
    tree = parser.raw_parse(sent)
    g = ParseTreeViz.buildVizGraph(tree, 'example', 'this is an example')
    fp = g.view()
