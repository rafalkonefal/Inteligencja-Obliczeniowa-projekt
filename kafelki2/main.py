import settings as s
import show_fcn as show
import ga


s.read_params()
results = []

if s.proceed_with_tiles:
    tiles = s.preinit_tile()
    for t, tile in enumerate(tiles):
        print("Tile",t)
        s.init(tile)
        results.append(ga.proceed())
    show.show_results(results)
else:
    s.init()
    show.showImage(s.img,'Original Img')
    results.append(ga.proceed())
    show.show_results(results)
