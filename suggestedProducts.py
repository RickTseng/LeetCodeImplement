def suggestedProducts(self, products: list[str], searchWord: str) -> list[list[str]]:
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