import pygame
import math
import random
import numpy as np

class floodfill:
    """
    Docstring for floodfill

    This is the "flood-fill" style maze solver.

    """
    def __init__(self, maze: np.ndarray, knownWalls: np.ndarray):
        """
        Docstring for __init__
        
        :param self: the class itself
        :param maze: This is the orginal matrix with a values indicate tha distance from that cell to the center (no diagonals)
        :type maze: np.ndarray
        :param knownWalls: a matrix the same size as the maz with dictionary in each cell to record the wall of that cell
        :type knownWalls: np.ndarray
        """
        self.maze = maze
        self.knownWalls = knownWalls
        self.penalty = np.zeros_like(self.maze) #resets every cycle
        self.opposite = {
            'n': 's',
            's': 'n',
            'w': 'e',
            'e': 'w'
        }

    def getNeighbours(self,row:int, col:int):
        """
        Docstring for getNeighbours

        This function gets the neighbours (no diagonals) if the given cell. 
        Returns a dictionary in the form {compass direction ('n','s','e','w') : cell position tuple}
        
        :param self: the class itself
        :param row: the row of the cell
        :type row: int
        :param col: the column of the cell
        :type col: int
        """
        allneighbors = {
            'e': (row, col + 1),
            'w': (row, col - 1),
            'n': (row - 1, col),
            's': (row + 1, col),
        }

        checkedneighbors = {
            compass: pair
            for compass, pair in allneighbors.items()
            if 0 <= pair[0] <= (((self.maze).shape)[0]-1)  and 0 <= pair[1] <= (((self.maze).shape)[1]-1)
        }
        return checkedneighbors

    def updateKnownWalls(self, targetCell:tuple, s1:bool, s2:bool, s3:bool, s4:bool, s5:bool, direction:int, currentRow:int, currentCol:int):
        """
        Docstring for updateKnownWalls

        This function updates the known walls based on the sensors input
        Note that for this sensor layout if s1,s2,s3 are True there is a wall directly infront if not then the sensor are detecting the cell infront.
        Returns nothing
        
        :param self: the class itself
        :param targetCell: this is the cell infront (for this model)
        :type targetCell: tuple
        :param s1: postive angle sensor
        :type s1: bool
        :param s2: center sensor
        :type s2: bool
        :param s3: negative angle sensor
        :type s3: bool
        :param s4: left sensor
        :type s4: bool
        :param s5: right sensor
        :type s5: bool
        :param direction: 0=east, 1=south, 2=west, 3=north
        :type direction: int
        :param currentRow: the current row
        :type currentRow: int
        :param currentCol: the current column
        :type currentCol: int
        """
        
        targetSensorMap = {
            0: {'s1': 'n','s2': 'e','s3': 's',},  # east
            1: {'s1': 'e','s2': 's','s3': 'w'},    # south
            2: {'s1': 's','s2': 'w','s3': 'n'},     # west
            3: {'s1': 'w','s2': 'n','s3': 'e'},   # north
        }

        currentSensorMap = {
            0: {'s4': 'n', 's5': 's'}, # east
            1: {'s4': 'e', 's5': 'w'}, # south
            2: {'s4': 's', 's5': 'n'}, # west
            3: {'s4': 'w', 's5': 'e'}, # north
        }

        sensorValues = {'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5}

        if s1 == True and s2 == True and s3 == True: 
            frontWall = targetSensorMap[direction]['s2']
            self.knownWalls[currentRow][currentCol][frontWall] = True
        else:
            for sensor, wallDir in targetSensorMap[direction].items():
                self.knownWalls[targetCell[0]][targetCell[1]][wallDir] = sensorValues[sensor]

        for sensor, wallDir in currentSensorMap[direction].items():
            self.knownWalls[currentRow][currentCol][wallDir] = sensorValues[sensor]

        for coor, walldict in np.ndenumerate(self.knownWalls):
            neighbours = self.getNeighbours(coor[0], coor[1])
            for wall, status in walldict.items():
                if not status:
                    continue
                if wall in neighbours:
                    neighbourRow, neighboursCol = neighbours[wall]
                else:
                    continue
                self.knownWalls[neighbourRow][neighboursCol][self.opposite[wall]] = True

        return

    def wallCount(self, row:int, col:int):
        """
        Docstring for wallCount
        This function count the walls of the given cell
        
        :param self: the class itself
        :param row: the row of the cell
        :type row: int
        :param col: the column of the cell
        :type col: int
        """

        return sum(self.knownWalls[row][col].values())

    def cellCost(self, cell:tuple):
        """
        Docstring for cellCost
        This function calulates the "cost" of the cell bu combining the distance with the "dead-end" penality
        Return a int

        :param self: the class itself
        :param cell: the given cell
        :type cell: tuple
        """
        row, col = cell
        return self.maze[row][col] + self.penalty[row][col]

    def nextMove(self,tempCell:tuple,tempCount:list,row:int, col:int, direction:int):
        """
        Docstring for nextMove

        This function calulates the next step to get to the goal
        Returns row, col, direction, tempCell, tempCount
        
        :param self: the class itself
        :param tempCell: the temporary cell to consider
        :type tempCell: tuple
        :param tempCount: list of all the cell visited
        :type tempCount: list
        :param row: the current cell's row
        :type row: int
        :param col: the current cell' column
        :type col: int
        :param direction: the current direction
        :type direction: int
        """
        nearTemp = self.getNeighbours(tempCell[0], tempCell[1])

        validNeighbours = [
            cell for compass, cell in nearTemp.items()
            if self.knownWalls[cell[0]][cell[1]][self.opposite[compass]] == False
        ]

        neighbourCount = {
            cell: self.maze[cell[0]][cell[1]]
            for cell in validNeighbours
        }

        maxBack = int(self.maze.max() / 100) * 100
        if maxBack < 1:
            maxBack = 100

        noPastNeighbour = {
            cell: val for cell, val in neighbourCount.items()
            if cell not in tempCount and self.maze[cell[0]][cell[1]] < maxBack
        }

        if noPastNeighbour == {}:
            if self.wallCount(tempCell[0], tempCell[1]) >= 3:
                self.penalty[tempCell[0]][tempCell[1]] += 100  
            else:
                self.penalty[tempCell[0]][tempCell[1]] += 50  

            neighbours = {cell: self.cellCost(cell) for cell in validNeighbours}  # ← USE cellCost
            minDistance = min(neighbours.values())
            minCell = random.choice(
                [cell for cell, val in neighbours.items() if val == minDistance]
            )
        else:
            costs = {cell: self.cellCost(cell) for cell in noPastNeighbour}  # ← USE cellCost
            minDistance = min(costs.values())
            minCell = random.choice(
                [cell for cell, val in costs.items() if val == minDistance]
            )

        tempCount.append(minCell)
        tempCell = minCell

        faceTempCell = next(
            (k for k, v in nearTemp.items() if v == tempCell),
            None
        )

        compassToDirect = {'e': 0, 's': 1, 'w': 2, 'n': 3}
        direction = compassToDirect[faceTempCell]
        row, col = tempCell

        return row, col, direction, tempCell, tempCount

    def shortestPath(self,path:list):
        """
        Docstring for shortestPath

        This updates the maze after a path to goal has been found. 
        Updates values based on the length of path.
        Returns nothing
        
        :param self: Description
        :param path: Description
        :type path: list
        """
        maxBack = int((self.maze).max()/100)*100
        if maxBack < 1:
            maxBack = 100
        backPath = [cell for cell in path if self.penalty[cell[0]][cell[1]] > maxBack]
        pastPath = [cell for cell in path if cell not in backPath]
        for cell in pastPath:
            pos = pastPath.index(cell) + 1
            self.maze[cell[0]][cell[1]] = len(pastPath) + 1 - pos
        return

class mouse: #will need to update foe real life 
    # direction: 0=east, 1=south, 2=west, 3=north
    """
    Docstring for mouse
    
    This will hold all function/control of the raspberypi mouse
    """

    def infrontCell(self,maze,knownWalls,currentrow, currentcol, direction):#as sensour for pygame can onlt detect in front
        """
        This is a place holder function. Current this vitual maze model onlt can sense the cell dicrectly in front. This maybe not be in the case in real life
        """
        find = floodfill(maze,knownWalls)
        neighbours = find.getNeighbours(currentrow, currentcol)

        if direction == 0:      # east
            return neighbours.get('e', (currentrow, currentcol))
        elif direction == 1:    # south
            return neighbours.get('s', (currentrow, currentcol))
        elif direction == 2:    # west
            return neighbours.get('w', (currentrow, currentcol))
        elif direction == 3:    # north
            return neighbours.get('n', (currentrow, currentcol))
        else:
            return (currentrow, currentcol)
        
#==========================Virutual Config===========================
CELL = 100
WALL = 8
GRID = 9
SCREEN = GRID * CELL

WHITE = (255,255,255)
BLACK = (0,0,0)
RED   = (255,0,0)
BLUE  = (0,0,255)
GREEN = (0,200,0)

pygame.init()
screen = pygame.display.set_mode((SCREEN, SCREEN))
clock = pygame.time.Clock()

# ================= MAZE =================
walls = [[[1,1,1,1] for _ in range(GRID)] for _ in range(GRID)]
visited = [[False]*GRID for _ in range(GRID)]

def carve(r,c):
    visited[r][c] = True
    dirs = [(0,1,1,3),(1,0,2,0),(0,-1,3,1),(-1,0,0,2)]
    random.shuffle(dirs)
    for dr,dc,w1,w2 in dirs:
        nr,nc = r+dr,c+dc
        if 0<=nr<GRID and 0<=nc<GRID and not visited[nr][nc]:
            walls[r][c][w1] = 0
            walls[nr][nc][w2] = 0
            carve(nr,nc)

carve(0,0)

# ================= REMOVE SOME WALLS =================
REMOVE_PROB = 0.05

for r in range(1, GRID-1):
    for c in range(1, GRID-1):
        for side in range(4):
            if walls[r][c][side] == 1 and random.random() < REMOVE_PROB:
                walls[r][c][side] = 0

                if side == 0 and r > 0:
                    walls[r-1][c][2] = 0
                elif side == 1 and c < GRID-1:
                    walls[r][c+1][3] = 0
                elif side == 2 and r < GRID-1:
                    walls[r+1][c][0] = 0
                elif side == 3 and c > 0:
                    walls[r][c-1][1] = 0

# ================= WALL RECTS =================
wall_rects = []
for r in range(GRID):
    for c in range(GRID):
        x,y = c*CELL, r*CELL
        t,rt,b,l = walls[r][c]
        if t:  wall_rects.append(pygame.Rect(x,y,CELL,WALL))
        if rt: wall_rects.append(pygame.Rect(x+CELL-WALL,y,WALL,CELL))
        if b:  wall_rects.append(pygame.Rect(x,y+CELL-WALL,CELL,WALL))
        if l:  wall_rects.append(pygame.Rect(x,y,WALL,CELL))

def draw_maze():
    for w in wall_rects:
        pygame.draw.rect(screen, BLACK, w)

# ================= RAY SENSOR =================
def cast_ray(origin, angle_deg, max_len):
    ang = math.radians(angle_deg)
    MIN_DIST = 8

    for d in range(MIN_DIST, max_len):
        x = int(origin[0] + math.cos(ang)*d)
        y = int(origin[1] + math.sin(ang)*d)

        if x < 0 or y < 0 or x >= SCREEN or y >= SCREEN:
            return (x,y), True

        for w in wall_rects:
            if w.collidepoint(x,y):
                return (x,y), True

    end = (
        origin[0] + math.cos(ang)*max_len,
        origin[1] + math.sin(ang)*max_len
    )
    return end, False

def dir_angle(d):
    return [0, 90, 180, -90][d]

last_states = {}

def report(sensor_name, hit):
    state = True if hit else False
    last_states[sensor_name] = hit
    return [sensor_name, state]

#============================================================

# ================= MOUSE Initals=================
row = 0
col = 8
direction = 0
#these will chanf in real life ^


UQmarsMaze = np.array([
    [9,8,7,6,5,6,7,8,9],
    [8,7,6,5,4,5,6,7,8],
    [7,6,5,4,3,4,5,6,7],
    [6,5,4,3,2,3,4,5,6],
    [5,4,3,2,1,2,3,4,5],
    [6,5,4,3,2,3,4,5,6],
    [7,6,5,4,3,4,5,6,7],
    [8,7,6,5,4,5,6,7,8],
    [9,8,7,6,5,6,7,8,9]
])
UQmarsMazeWalls = np.array([
            [
                {'n': False, 's': False, 'w': False, 'e': False}
                for _ in range((UQmarsMaze).shape[1])
            ]
            for _ in range((UQmarsMaze).shape[0]) ])


micromouse = floodfill(UQmarsMaze,UQmarsMazeWalls)

pimouse = mouse()

tempcount = [(row,col)]
tempcell = (row,col)


# ================= MAIN LOOP =================
running = True
while running:
    #==========PYGAME===========
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)
    pygame.draw.rect(screen, GREEN, (4*CELL+WALL, 4*CELL+WALL, CELL-2*WALL, CELL-2*WALL))

    cx = col*CELL + CELL//2
    cy = row*CELL + CELL//2
    center = (cx, cy)

    base = dir_angle(direction)

    named_sensors = [
        ("CENTER", 0, GREEN),
        ("DIAG_UP", -30, BLUE),
        ("DIAG_DOWN", 30, BLUE),
        ("PERP_UP", 90, BLUE),
        ("PERP_DOWN", -90, BLUE),
    ]

    sense = {}
    for name, off, default_color in named_sensors:
        end, hit = cast_ray(center, base + off, 120)
        color = RED if hit else default_color
        pygame.draw.line(screen, color, center, end, 4)
        info = report(name, hit)
        sense[info[0]] = info[1]

    pygame.draw.rect(screen, RED, (cx-8, cy-8, 16, 16))
    draw_maze()
    pygame.display.flip()
    clock.tick(6)

    #====================MicroMouse=============================
    if micromouse.maze[row][col] != 1:
        cellinfront = pimouse.infrontCell(UQmarsMaze,UQmarsMazeWalls,row, col, direction)
        micromouse.updateKnownWalls(cellinfront,
            sense["DIAG_UP"], sense["CENTER"], sense["DIAG_DOWN"],
            sense["PERP_DOWN"], sense["PERP_UP"],
            direction, row, col)
        
        row,col,direction,tempcell,tempcount  = micromouse.nextMove(tempcell,tempcount,row, col, direction)

    else:
        print(micromouse.maze)
        row = 0
        col = 8
        direction = 0
        micromouse.shortestPath(tempcount)
        tempcount = [(row,col)]
        tempcell = (row,col)

pygame.quit()