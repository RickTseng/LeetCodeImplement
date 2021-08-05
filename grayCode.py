def grayCode(self, n: int) -> list[int]:
        res = list()
        para = list([0,1])
        for i in range(n):
            for p in range(len(para)):
                if i != 0:                
                    para.append(para[p]+pow(2,i))
                    res.append(para[p]+pow(2,i))
                else:
                    res.append(para[p])
            l = len(res) -1
            para.clear()
            while l >-1:
                para.append(res[l])
                l-=1
        return res