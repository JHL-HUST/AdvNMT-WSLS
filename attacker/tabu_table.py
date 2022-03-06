import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

class tabu_list():

    def __init__(self,size):
        self.size=size
        self.tabu_dic = {}

    def put(self,ele):
        if self.query(ele):
            raise Exception('put an element which is already in the tabu table')
        else:
            self.tabu_iter()
            self.tabu_dic[ele]=1

    def tabu_iter(self):
        remove_list = []
        for key in self.tabu_dic.keys():
            self.tabu_dic[key] = self.tabu_dic[key] + 1
            if self.tabu_dic[key] > self.size:
                remove_list.append(key)
        
        for key in remove_list:
            self.tabu_dic.pop(key)
        

    def show(self):
        print(self.tabu_dic)
    def query(self,ele):
        if self.tabu_dic.__contains__(ele):
            return True
        else:
            return False

if __name__ == "__main__":
    q=tabu_list(2)
    q.put(0)
    q.put(1)

    q.show()
    q.put(3)
    q.show()
    q.put(4)
    q.show()

    print(q.query(3))
    print(q.query(0))
