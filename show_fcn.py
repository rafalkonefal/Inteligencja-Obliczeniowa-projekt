from matplotlib import pyplot as plt
import numpy as np
import settings as s
from kafelki import compose_img


def showImage(img, title = ''):
    plt.figure()
    plt.imshow(img,cmap='gray')
    plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.show()
    

def show_results(results):
    if len(results) == 1:
        plt.figure()
        plt.plot(results[0][0])
        plt.xlim(0, s.ga_maxit)
        plt.xlabel('Iterations')
        plt.ylabel('Best Cost')
        plt.title('Genetic Algorithm (GA)')
        plt.grid(True)
        plt.show()
        showImage(results[0][1],'Genetic Algorithm')
    else:
        # kafelki
        tiles = [t[1] for t in results]
        img = compose_img(tiles, s.img_size, s.tile_size, s.shift)
        showImage(img,'Genetical Algorithm')
        costs = [t[0] for t in results]
        costs_list = [[] for _ in range(s.ga_maxit)]
        for t,tile_cost in enumerate(costs):
            for it,cost in enumerate(tile_cost):
                costs_list[it].append(cost)
            if not s.show_all_plots:
                continue
            plt.figure()
            plt.plot(tile_cost)
            plt.xlim(0, s.ga_maxit)
            plt.xlabel('Iterations')
            plt.ylabel('Best Cost')
            plt.title('Genetic Algorithm (GA) - Tile '+str(t))
            plt.grid(True)
            plt.show()
                
        plt.figure()
        plt.plot(costs_list)
        plt.xlim(0, s.ga_maxit)
        plt.xlabel('Iterations')
        plt.ylabel('Best Cost')
        plt.title('Genetic Algorithm (GA) - All Tiles')
        plt.grid(True)
        plt.show()
        
        costs_list = [np.mean(it) for it in costs_list]
        plt.figure()
        plt.plot(costs_list)
        plt.xlim(0, s.ga_maxit)
        plt.xlabel('Iterations')
        plt.ylabel('Mean Best Cost')
        plt.title('Genetic Algorithm (GA)')
        plt.grid(True)
        plt.show()

    
    
    
    
    
    
    
    
    
    
    