def searchRange(self, nums: list[int], target: int) -> list[int]:      
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