from typing import List
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        base = []
        total = len(nums)
        item = 0
        for i in range(len(nums)):
            base.append([i])           
        while item < total-1:
            newBase = []
            for i in range(len(base)):
                
                for j in range(len(nums)):
                    
                    if j != base[i][item]:
                        tmp =[]
                        restmp = []
                        for k in range(len(base[i])):
                            tmp.append(base[i][k])
                        if j not in tmp:
                            tmp.append(j)
                        

                            if restmp not in res:
                                newBase.append(tmp)                            
            base = newBase
            item+=1
        for i in range(len(base)):
            for j in range(len(base[i])):
                base[i][j] = nums[base[i][j]]
        return base
