class HumanBeing(object):
      def __init__(self, name, eye, position):
          self.name = name
          self.coloreye = eye
          self.position = position
      def walksteps(self, steps):
          self.position += steps

Bill = HumanBeing('Bill', 'blue', 10)
print(Bill.name)
print(Bill.coloreye)

Bill.walksteps(5)
print(Bill.position)
