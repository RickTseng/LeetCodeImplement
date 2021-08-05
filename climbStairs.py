class Solution:
    def factory(n:int)->int:
        res = 1
        for i in range(n):
            res *= i+1

        return res
    def climbStairs(self, n: int) -> int:
        count = 0
        twoStep = int(n/2)
        for i in range(twoStep +1):
            if i ==0:
                count+=1
            else:
                totalStep = i*2
                oneStep = 0
                while totalStep != n:
                    totalStep+=1
                    oneStep+=1

                count += int(Solution.factory(i+oneStep)/ (Solution.factory(oneStep) * Solution.factory(i)))
        return count