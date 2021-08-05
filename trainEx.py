
from os import lseek, system
from typing import List, Tuple
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right    
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        n1 = list(num1)
        n2 = list(num2)
        tmp  = list(list())
        for i in range(len(n1)):
            L1 = list()
            for j in range(len(n2)):
                L1.append(int(n1[i])*int(n2[j]))
            tmp.append(L1)
        res = list()
        for i in range(len(tmp)):
            count = 0
            for j in range(i +1):
                
                count+=tmp[i][j]
                
            res.append(count)
        return

    
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i == 0 or (i > 0 and nums[i] != nums[i-1]):
                j, k, target = i + 1, len(nums) - 1, 0 - nums[i]
                while j < k:
                    if nums[j] + nums[k] == target:
                        res.append([nums[i], nums[j], nums[k]])
                        while j < k and nums[j] == nums[j+1]:
                            j += 1
                        while j < k and nums[k] == nums[k-1]:
                            k -= 1
                        j += 1
                        k -= 1
                    elif nums[j] + nums[k] < target:
                        j += 1
                    else:
                        k -= 1
        return res
    def lengthOfLastWord(self, s: str) -> int:
        x = len(list(s))-1
        count = 0
        while x >-1:
            if(s[x]!=' '):
                count+=1
            else:
                if(count!=0):
                    break

            x-=1
        return count
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        return res 
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        return
    def maxSubArray(self, nums: List[int]) -> int:

        

        return
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
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return x
        low = 1
        high = x
        midd = int(0)
        while low <= high:
            midd = int(low + (high -low) / 2)
            if midd > x /midd:
                high = midd -1
            elif midd<x/midd:
                low = midd+1
            else:
                return int(midd)

        return int(high)
    def buildLN(val:int,org:ListNode)->ListNode:
        rtn = ListNode()
        org.val = val
        rtn.next = org
        return rtn
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None:
            return head
        valList =list()
        ll = head
        while ll!=None:
            valList.append(ll.val)
            ll = ll.next
        valList = list(set(valList))
        valList.sort()
        res = ListNode()
        i = len(valList)-1
        while i>-1:
            if i==0:
                res.val = valList[i]
            else:
                res = Solution.buildLN(valList[i],res)
            i-=1  
        return res
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        for i in range(n):
            nums1.insert(m+i,nums2[i])
            nums1.pop(len(nums1)-1)

        nums1.sort()

        return
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        x= accounts[0]
        


        return
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        
        
        start = 0
        end = len(nums) -1
        Max = nums[end]
        Min = nums[start]
        while start<=end:
            mid = int(start + (end -start) / 2)
            if nums[mid] > target :
                end = mid -1
            elif nums[mid] < target :
                start = mid+1
            else:
                break
        if start>end:
            return [-1,-1]
        while nums[end]>target:
            end -=1        
        while mid>0 and nums[mid-1]>=target:
            mid -=1     
                
        
        return [mid,end]
    def grayCode(self, n: int) -> List[int]:
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

    def partitionDisjoint(self, nums: List[int]) -> int:
        
        return
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        com = target
        cIndexLen = len(candidates) -1
        
        while cIndexLen >-1:
            oneType = []
            com = target
            cSubIndexLen = cIndexLen
            while com !=0 and cSubIndexLen>-1:
                if com >candidates[cSubIndexLen]:
                    oneType.append(candidates[cSubIndexLen])
                    com -=candidates[cSubIndexLen]
                elif com == candidates[cSubIndexLen]:
                    oneType.append(candidates[cSubIndexLen])
                    com -=candidates[cSubIndexLen]
                    res.append(oneType)
                    break
                else:
                    #com += candidates[cSubIndexLen]
                
                    cSubIndexLen-=1
                    if com < candidates[cSubIndexLen]:
                        com += candidates[cSubIndexLen]
                    
                        
            cIndexLen -=1
        return res
    def toZero(matrix: List[List[int]], row:int,col:int) ->List[List[int]]:
        for r in range(len(matrix)):
            for c in range(len(matrix[row])):
                if r == row or c == col:
                    matrix[r][c] = 0
                    
        return matrix
    def setZeroes(self, matrix: List[List[int]]) -> None:
        row = len(matrix)
        col = len(matrix[0])
        point = list()
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                if matrix[row][col] == 0:
                    point.append([row,col])
                    #matrix = Solution.toZero(matrix,row,col)

        for p in range(len(point)):
            matrix = Solution.toZero(matrix,point[p][0],point[p][1])

        return None
    def beautifulArray(self, n: int) -> List[int]:
        res = []
        res.append(1)
        while len(res) < n:
            tmp = []
            for i in range(len(res)):
                if res[i]*2-1 > 0 and res[i] * 2 - 1 <= n:
                    tmp.append(res[i] * 2 - 1)
            for j in range(len(res)):
                if res[j]*2 > 0 and res[j]* 2 <= n:
                    tmp.append(res[j] * 2)
            if tmp !=[]:
                res = tmp

        return res
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        TotalCount = len(matrix) * len(matrix[0])
        pi = 0
        pj = 0
        count = 0
        rowStart = 0
        colStart = 0
        rowEnd = len(matrix[0])
        colEnd = len(matrix)
        
        while count < TotalCount:
            while pj < rowEnd and count < TotalCount:
                res.append(matrix[pi][pj])
                pj+=1
                count+=1
            colStart+=1
            rowEnd -=1
            pj = rowEnd
            pi+=1
            while pi < colEnd and count < TotalCount:
                res.append(matrix[pi][pj])
                pi +=1
                count+=1
            colEnd -=1
            pi = colEnd
            pj-=1
            while pj >= rowStart and count < TotalCount:
                res.append(matrix[pi][pj])
                pj -=1
                count+=1
            pj = rowStart
            pi-=1
            while pi >= colStart and count < TotalCount:
                res.append(matrix[pi][pj])
                pi -=1
                count+=1
            rowStart+=1
            pi = rowStart
            pj+=1
        return res
    def generateMatrix(self, n: int) -> List[List[int]]:
        res = []
        for r in range(n):
            clist = []
            for c in range(n):
                clist.append(0)
            res.append(clist)

        TotalCount = n * n
        pi = 0
        pj = 0
        count = 1
        rowStart = 0
        colStart = 0
        rowEnd = n
        colEnd = n
        
        while count <= TotalCount:
            while pj < rowEnd and count <= TotalCount:
                res[pi][pj] = count
                pj+=1
                count+=1
            colStart+=1
            rowEnd -=1
            pj = rowEnd
            pi+=1
            while pi < colEnd and count <= TotalCount:
                res[pi][pj] = count
                pi +=1
                count+=1
            colEnd -=1
            pi = colEnd
            pj-=1
            while pj >= rowStart and count <= TotalCount:
                res[pi][pj] = count
                pj -=1
                count+=1
            pj = rowStart
            pi-=1
            while pi >= colStart and count <= TotalCount:
                res[pi][pj] = count
                pi -=1
                count+=1
            rowStart+=1
            pi = rowStart
            pj+=1
        return res
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
        res = []
        res.append([rStart,cStart])
        moveStep = 1
        step = 0
        rLimit = rows
        cLimit = cols
        pi = rStart 
        pj = cStart + 1
        TotalCount = rows * cols
        count = 1
        while count < TotalCount:
            while step < moveStep and count < TotalCount:
                if pj< cLimit and pi< rLimit and pj >-1 and pi >-1:
                    res.append([pi,pj])
                    count+=1
                step+=1
                pj +=1
            pj-=1
            pi+=1
            step = 0
            while step < moveStep and count < TotalCount:
                if pj< cLimit and pi< rLimit and pj >-1 and pi >-1:
                    res.append([pi,pj])
                    count+=1
                step+=1
                pi +=1
            moveStep+=1
            pj-=1
            pi-=1
            step = 0
            while step < moveStep and count < TotalCount:
                if pj< cLimit and pi< rLimit and pj >-1 and pi >-1:
                    res.append([pi,pj])
                    count+=1
                step+=1
                pj -=1
            pj+=1
            pi-=1
            step = 0
            while step < moveStep and count < TotalCount:
                if pj< cLimit and pi< rLimit and pj >-1 and pi >-1:
                    res.append([pi,pj])
                    count+=1
                step+=1
                pi-=1
            moveStep+=1
            pi+=1
            pj+=1
            step = 0
        return res
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        import sys
        res =[]
        for row in range(len(mat)):
            tmpRow = []
            for col in range(len(mat[row])):
                tmpRow.append(sys.maxsize -1)
            res.append(tmpRow)
        for row in range(len(mat)):
            for col in range(len(mat[row])):
                if mat[row][col] == 1:
                    if row > 0 :
                        res[row][col] = min(res[row][col],res[row-1][col] +1)
                    if col > 0 :
                        res[row][col] = min(res[row][col],res[row][col-1]+1)
                else:
                    res[row][col] = 0
        for row in range(len(mat)):
            for col in range(len(mat[row])):
                rowRev = len(mat) - row - 1
                colRev = len(mat[row]) - col - 1
                if rowRev < len(mat) -1:
                    res[rowRev][colRev] = min(res[rowRev][colRev],res[rowRev+1][colRev] +1)
                if colRev < len(mat[row]) -1:
                    res[rowRev][colRev] = min(res[rowRev][colRev],res[rowRev][colRev+1]+1)
                    
                


        return res
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res = []
        get = False
        for i in range(len(nums)):
            for j in range(len(nums)-1,-1,-1):
                if i == j:
                    break
                if nums[i] + nums[j] == target:
                    res.append(i)
                    res.append(j)
                    get = True
                    break
                else:
                    get = False
                    
                
            if get:
                break
        return res
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        i= 0
        get = False
        while i < len(intervals):
            tmp = []
            if intervals[i][1]<newInterval[0] or intervals[i][0]>newInterval[1]:
                if intervals[i][0]>newInterval[1] and not get:
                    res.append(newInterval)
                    get = True
                res.append(intervals[i])
            
            else:
                tmp.append(min(intervals[i][0],newInterval[0]))
                
                
                while i<len(intervals):
                    if i+1<len(intervals) and intervals[i][1]<= newInterval[1] and newInterval[1]< intervals[i+1][0]:
                        tmp.append(newInterval[1])
                        res.append(tmp)
                        get = True
                        break
                    if intervals[i][1]>=newInterval[1] and not get:
                        tmp.append(max(intervals[i][1],newInterval[1]))
                        res.append(tmp)
                        get = True
                        break
                    else:
                        i+=1
                if not get:
                    tmp.append(newInterval[1])
                    res.append(tmp)
                    get = True
                    break
            i+=1
        if not get:
            res.append(newInterval)

        return res
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        base = []
        res.append([])
        total = len(nums)
        item = 0
        for i in range(len(nums)):
            base.append([nums[i]])
            res.append([nums[i]])
        while item < total-1:
            newBase = []
            for i in range(len(base)):
                
                for j in range(len(nums)):
                    
                    if nums[j]> base[i][item]:
                        tmp =[]
                        for k in range(len(base[i])):
                            tmp.append(base[i][k])
                        tmp.append(nums[j])
                        newBase.append(tmp)
                        res.append(tmp)

            base = newBase
            item+=1
        return res
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        base = []
        res.append([])
        total = len(nums)
        item = 0
        for i in range(len(nums)):
            base.append([i])
            if [nums[i]] not in res:
                res.append([nums[i]])
        while item < total-1:
            newBase = []
            for i in range(len(base)):
                
                for j in range(len(nums)):
                    
                    if j > base[i][item]:
                        tmp =[]
                        restmp = []
                        for k in range(len(base[i])):
                            tmp.append(base[i][k])
                            restmp.append(nums[base[i][k]])
                        tmp.append(j)
                        restmp.append(nums[j])
                        restmp.sort()
                        if restmp not in res:
                            newBase.append(tmp)
                            res.append(restmp)

            base = newBase
            item+=1
        return res

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        startIndex = -1
        for i in range(len(gas)):
            stationCount = 0
            oil = 0
            startIndex = i
            while stationCount < len(gas):
                oil += gas[i]
                if oil - cost[i] >=0:
                    oil -=cost[i]
                    stationCount+=1
                    i+=1
                    if i> len(gas)-1:
                        i -=len(gas)
                else:
                    break
            if stationCount == len(gas):
                break
        if stationCount != len(gas):
            startIndex = -1

        return startIndex
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        ans = []  
        sortProduct = []     
        for p in range(len(products)):
            sortProduct.append(products[p])
        sortProduct.sort()
        for i in range(len(searchWord)):
            tmp = []
            toAns = []
            for j in range(len(sortProduct)):
                if i< len(sortProduct[j]) and sortProduct[j][i] == searchWord[i]:
                    tmp.append(sortProduct[j])
                    if len(toAns)<3:
                        toAns.append(sortProduct[j])
                    
            sortProduct = tmp
            
            ans.append(toAns)
        return ans
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
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
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
                            restmp.append(nums[base[i][k]])
                        if j not in tmp:
                            tmp.append(j) 
                            restmp.append(nums[j])                     
                            if restmp not in res:
                                newBase.append(tmp)                            
            base = newBase
            item+=1
        for i in range(len(base)):
            for j in range(len(base[i])):
                base[i][j] = nums[base[i][j]]
            if base[i] not in res:
                res.append(base[i])  
        return res
  
        
        
                
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        res, queue = [],[(root,'')]
        while queue:
            node,curs = queue.pop()
            if node:
                if not node.left and not node.right:
                    res.append(curs + str(node.val))
                queue.insert(0,(node.left,curs + str(node.val)+'->'))
                queue.insert(0,(node.right,curs + str(node.val)+'->'))
        return res
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        res,path_list = [],[]
        self.dfs(root,path_list,res,targetSum)
        return res
    def dfs( root:TreeNode,path_list:list,res:list,target:int):
        if not root:
            return
        path_list.append(root.val)
        if not root.left and not root.right:
            Sum = 0
            for i in range(len(path_list)):
                Sum+=path_list[i]
            if Sum == target:
                res.append(list(path_list))
        if root.left:
            Solution.dfs(root.left,path_list,res,target)
        if root.right:
            Solution.dfs(root.right,path_list,res,target)
        path_list.pop()
    def binaryTreePath2(self, root: TreeNode) -> List[str]:
        res,path_list = [],[]
        self.dfs(root,path_list,res)

        return res

tree = TreeNode(1)
tree.left = TreeNode(2)
tree.right = TreeNode(3)
tree.left.right = TreeNode(5)
#z = Solution.binaryTreePath2(Solution,root=tree)
x = Solution.pathSum(Solution,root=tree,targetSum=8)

y=1
