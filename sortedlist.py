class SortedList(list):
    def __getitem__(self, key):
        return super(SortedList, self).__getitem__(key)
    @staticmethod
    def fromlist(arr):
        out = SortedList()
        arrs = sorted(arr)
        for a in arrs:
            out.append(a)
        return out
    def index(self, val):
        startpt = 0
        endpt = self.__len__()-1
        while True:
            midpt = (startpt + endpt) // 2
            midval = self.__getitem__(midpt)
            if startpt == midpt and midval != val:
                break
            if val < midpt:
                endpt = midpt - 1
            elif val > midpt:
                startpt = midpt + 1
            else:
                return midpt
        return midpt
    def range_index(self, lower, upper):
        return self.index(lower), self.index(upper)