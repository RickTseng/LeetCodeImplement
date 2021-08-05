def canCompleteCircuit(self, gas: list[int], cost: list[int]) -> int:
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