#!/usr/bin/python
#-*- coding:utf-8 -*-

from __future__ import division

import itertools
import cPickle
import datetime
import hashlib
import locale
import numpy as np
import pycountry
import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd
from collections import defaultdict
from sklearn.preprocessing import normalize

#封装数据清洗方法
class DataCleaner:
  """
  Common utilities for converting strings to equivalent numbers
  or number buckets.
  """
  def __init__(self):
    # 载入 locales
    self.localeIdMap = defaultdict(int)
    for i, l in enumerate(locale.locale_alias.keys()):
      self.localeIdMap[l] = i + 1
    # 载入 countries
    self.countryIdMap = defaultdict(int)
    ctryIdx = defaultdict(int)
    for i, c in enumerate(pycountry.countries):
      self.countryIdMap[c.name.lower()] = i + 1
      if c.name.lower() == "usa":
        ctryIdx["US"] = i
      if c.name.lower() == "canada":
        ctryIdx["CA"] = i
    for cc in ctryIdx.keys():
      for s in pycountry.subdivisions.get(country_code=cc):
        self.countryIdMap[s.name.lower()] = ctryIdx[cc] + 1
    # 载入 gender id 字典
    self.genderIdMap = defaultdict(int, {"male":1, "female":2})

  def getLocaleId(self, locstr):
    return self.localeIdMap[locstr.lower()]

  def getGenderId(self, genderStr):
    return self.genderIdMap[genderStr]

  def getJoinedYearMonth(self, dateString):
    dttm = datetime.datetime.strptime(dateString, "%Y-%m-%dT%H:%M:%S.%fZ")
    return "".join([str(dttm.year), str(dttm.month)])

  def getCountryId(self, location):
    if (isinstance(location, str)
        and len(location.strip()) > 0
        and location.rfind("  ") > -1):
      return self.countryIdMap[location[location.rindex("  ") + 2:].lower()]
    else:
      return 0

  def getBirthYearInt(self, birthYear):
    try:
      return 0 if birthYear == "None" else int(birthYear)
    except:
      return 0

  def getTimezoneInt(self, timezone):
    try:
      return int(timezone)
    except:
      return 0

  def getFeatureHash(self, value):
    if len(value.strip()) == 0:
      return -1
    else:
      return int(hashlib.sha224(value).hexdigest()[0:4], 16)

  def getFloatValue(self, value):
    if len(value.strip()) == 0:
      return 0.0
    else:
      return float(value)

#user与event关联数据处理
class ProgramEntities:
  def __init__(self):
    # 统计训练集中有多少独立的用户的events
    uniqueUsers = set()
    uniqueEvents = set()
    eventsForUser = defaultdict(set)
    usersForEvent = defaultdict(set)
    for filename in ["train.csv", "test.csv"]:
      f = open(filename, 'rb')
      f.readline().strip().split(",")
      for line in f:
        cols = line.strip().split(",")
        uniqueUsers.add(cols[0])
        uniqueEvents.add(cols[1])
        eventsForUser[cols[0]].add(cols[1])
        usersForEvent[cols[1]].add(cols[0])
      f.close()
    self.userEventScores = ss.dok_matrix((len(uniqueUsers), len(uniqueEvents)))
    self.userIndex = dict()
    self.eventIndex = dict()
    for i, u in enumerate(uniqueUsers):
      self.userIndex[u] = i
    for i, e in enumerate(uniqueEvents):
      self.eventIndex[e] = i
    ftrain = open("train.csv", 'rb')
    ftrain.readline()
    for line in ftrain:
      cols = line.strip().split(",")
      i = self.userIndex[cols[0]]
      j = self.eventIndex[cols[1]]
      self.userEventScores[i, j] = int(cols[4]) - int(cols[5])
    ftrain.close()
    sio.mmwrite("PE_userEventScores", self.userEventScores)
    # 为了防止不必要的计算，找出来所有关联的用户或者关联的event
    self.uniqueUserPairs = set()
    self.uniqueEventPairs = set()
    for event in uniqueEvents:
      users = usersForEvent[event]
      if len(users) > 2:
        self.uniqueUserPairs.update(itertools.combinations(users, 2))
    for user in uniqueUsers:
      events = eventsForUser[user]
      if len(events) > 2:
        self.uniqueEventPairs.update(itertools.combinations(events, 2))
    cPickle.dump(self.userIndex, open("PE_userIndex.pkl", 'wb'))
    cPickle.dump(self.eventIndex, open("PE_eventIndex.pkl", 'wb'))

#建立用户与用户之间相似度矩阵
class Users:
  def __init__(self, programEntities, sim=ssd.correlation):
    cleaner = DataCleaner()
    nusers = len(programEntities.userIndex.keys())
    fin = open("users.csv", 'rb')
    colnames = fin.readline().strip().split(",")
    self.userMatrix = ss.dok_matrix((nusers, len(colnames) - 1))
    for line in fin:
      cols = line.strip().split(",")
      # 只考虑train.csv中出现的用户
      if programEntities.userIndex.has_key(cols[0]):
        i = programEntities.userIndex[cols[0]]
        self.userMatrix[i, 0] = cleaner.getLocaleId(cols[1])
        self.userMatrix[i, 1] = cleaner.getBirthYearInt(cols[2])
        self.userMatrix[i, 2] = cleaner.getGenderId(cols[3])
        self.userMatrix[i, 3] = cleaner.getJoinedYearMonth(cols[4])
        self.userMatrix[i, 4] = cleaner.getCountryId(cols[5])
        self.userMatrix[i, 5] = cleaner.getTimezoneInt(cols[6])
    fin.close()
    # 归一化用户矩阵
    self.userMatrix = normalize(self.userMatrix, norm="l1", axis=0, copy=False)
    sio.mmwrite("US_userMatrix", self.userMatrix)
    # 计算用户相似度矩阵
    self.userSimMatrix = ss.dok_matrix((nusers, nusers))
    for i in range(0, nusers):
      self.userSimMatrix[i, i] = 1.0
    for u1, u2 in programEntities.uniqueUserPairs:
      i = programEntities.userIndex[u1]
      j = programEntities.userIndex[u2]
      if not self.userSimMatrix.has_key((i, j)):
        usim = sim(self.userMatrix.getrow(i).todense(),
          self.userMatrix.getrow(j).todense())
        self.userSimMatrix[i, j] = usim
        self.userSimMatrix[j, i] = usim
    sio.mmwrite("US_userSimMatrix", self.userSimMatrix)

#对用户社交关系的挖掘
class UserFriends:
  def __init__(self, programEntities):
    nusers = len(programEntities.userIndex.keys())
    self.numFriends = np.zeros((nusers))
    self.userFriends = ss.dok_matrix((nusers, nusers))
    fin = open("user_friends.csv", 'rb')
    fin.readline()                # skip header
    ln = 0
    for line in fin:
      if ln % 200 == 0:
        print "Loading line: ", ln
      cols = line.strip().split(",")
      user = cols[0]
      if programEntities.userIndex.has_key(user):
        friends = cols[1].split(" ")
        i = programEntities.userIndex[user]
        self.numFriends[i] = len(friends)
        for friend in friends:
          if programEntities.userIndex.has_key(friend):
            j = programEntities.userIndex[friend]
            # the objective of this score is to infer the degree to
            # and direction in which this friend will influence the
            # user's decision, so we sum the user/event score for
            # this user across all training events.
            eventsForUser = programEntities.userEventScores.getrow(j).todense()
            score = eventsForUser.sum() / np.shape(eventsForUser)[1]
            self.userFriends[i, j] += score
            self.userFriends[j, i] += score
      ln += 1
    fin.close()
    # 归一化数组
    sumNumFriends = self.numFriends.sum(axis=0)
    self.numFriends = self.numFriends / sumNumFriends
    sio.mmwrite("UF_numFriends", np.matrix(self.numFriends))
    self.userFriends = normalize(self.userFriends, norm="l1", axis=0, copy=False)
    sio.mmwrite("UF_userFriends", self.userFriends)

#构造event和event之间相似度数据
class Events:
  def __init__(self, programEntities, psim=ssd.correlation, csim=ssd.cosine):
    cleaner = DataCleaner()
    fin = open("events.csv", 'rb')
    fin.readline() # skip header
    nevents = len(programEntities.eventIndex.keys())
    self.eventPropMatrix = ss.dok_matrix((nevents, 7))
    self.eventContMatrix = ss.dok_matrix((nevents, 100))
    ln = 0
    for line in fin.readlines():
#      if ln > 10:
#        break
      cols = line.strip().split(",")
      eventId = cols[0]
      if programEntities.eventIndex.has_key(eventId):
        i = programEntities.eventIndex[eventId]
        self.eventPropMatrix[i, 0] = cleaner.getJoinedYearMonth(cols[2]) # start_time
        self.eventPropMatrix[i, 1] = cleaner.getFeatureHash(cols[3]) # city
        self.eventPropMatrix[i, 2] = cleaner.getFeatureHash(cols[4]) # state
        self.eventPropMatrix[i, 3] = cleaner.getFeatureHash(cols[5]) # zip
        self.eventPropMatrix[i, 4] = cleaner.getFeatureHash(cols[6]) # country
        self.eventPropMatrix[i, 5] = cleaner.getFloatValue(cols[7]) # lat
        self.eventPropMatrix[i, 6] = cleaner.getFloatValue(cols[8]) # lon
        for j in range(9, 109):
          self.eventContMatrix[i, j-9] = cols[j]
        ln += 1
    fin.close()
    self.eventPropMatrix = normalize(self.eventPropMatrix,
        norm="l1", axis=0, copy=False)
    sio.mmwrite("EV_eventPropMatrix", self.eventPropMatrix)
    self.eventContMatrix = normalize(self.eventContMatrix,
        norm="l1", axis=0, copy=False)
    sio.mmwrite("EV_eventContMatrix", self.eventContMatrix)
    # calculate similarity between event pairs based on the two matrices
    self.eventPropSim = ss.dok_matrix((nevents, nevents))
    self.eventContSim = ss.dok_matrix((nevents, nevents))
    for e1, e2 in programEntities.uniqueEventPairs:
      i = programEntities.eventIndex[e1]
      j = programEntities.eventIndex[e2]
      if not self.eventPropSim.has_key((i,j)):
        epsim = psim(self.eventPropMatrix.getrow(i).todense(),
          self.eventPropMatrix.getrow(j).todense())
        self.eventPropSim[i, j] = epsim
        self.eventPropSim[j, i] = epsim
      if not self.eventContSim.has_key((i,j)):
        ecsim = csim(self.eventContMatrix.getrow(i).todense(),
          self.eventContMatrix.getrow(j).todense())
        self.eventContSim[i, j] = epsim
        self.eventContSim[j, i] = epsim
    sio.mmwrite("EV_eventPropSim", self.eventPropSim)
    sio.mmwrite("EV_eventContSim", self.eventContSim)

#event热度数据的处理
class EventAttendees():
  """
  统计某个活动参加和不参加的人数
  """
  def __init__(self, programEvents):
    nevents = len(programEvents.eventIndex.keys())
    self.eventPopularity = ss.dok_matrix((nevents, 1))
    f = open("event_attendees.csv", 'rb')
    f.readline() # skip header
    for line in f:
      cols = line.strip().split(",")
      eventId = cols[0]
      if programEvents.eventIndex.has_key(eventId):
        i = programEvents.eventIndex[eventId]
        self.eventPopularity[i, 0] = \
          len(cols[1].split(" ")) - len(cols[4].split(" "))
    f.close()
    self.eventPopularity = normalize(self.eventPopularity, norm="l1",
      axis=0, copy=False)
    sio.mmwrite("EA_eventPopularity", self.eventPopularity)

def data_prepare():
  """
  计算生成所有的数据，用矩阵或者其他形式存储方便提取特征和建模
  """
  print "第1步：统计user和event相关信息..."
  pe = ProgramEntities()
  print "第1步完成...\n"
  print "第2步：计算用户相似度信息，并用矩阵形式存储..."
  Users(pe)
  print "第2步完成...\n"
  print "第3步：计算用户社交关系信息，并存储..."
  UserFriends(pe)
  print "第3步完成...\n"
  print "第4步：计算event相似度信息，并用矩阵形式存储..."
  Events(pe)
  print "第4步完成...\n"
  print "第5步：计算event热度信息..."
  EventAttendees(pe)
  print "第5步完成...\n"

# 运行进行数据准备
data_prepare()