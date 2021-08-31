{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "c253f614",
   "metadata": {},
   "outputs": [],
   "source": [
    "vehicles_smallset_name = []\n",
    "non_vehicles_smallset_name = []\n",
    "\n",
    "for i in range(1, 4):\n",
    "    file_list = os.listdir('./vehicles_smallset/vehicles_smallset/cars{}'.format(i))\n",
    "\n",
    "    for file in file_list:\n",
    "        name = file.split('.')[0]\n",
    "        vehicles_smallset_name.append(name)\n",
    "    \n",
    "for i in range(1, 4):\n",
    "    file_list = os.listdir('./non-vehicles_smallset/non-vehicles_smallset/notcars{}'.format(i))\n",
    "    \n",
    "    for file in file_list:\n",
    "        name = file.split('.')[0]\n",
    "        non_vehicles_smallset_name.append(name)\n",
    "\n",
    "smallset_name = vehicles_smallset_name + non_vehicles_smallset_name\n",
    "smallset_num = []\n",
    "\n",
    "for num in range(1196):\n",
    "    smallset_num.append(1)\n",
    "\n",
    "for num in range(1125):\n",
    "    smallset_num.append(0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "4d11eafa",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1196"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(vehicles_smallset_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "d08fd975",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1125"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(non_vehicles_smallset_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "a89f4e61",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2321"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(smallset_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "9d36a93f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2321"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(smallset_num)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47c733d9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
