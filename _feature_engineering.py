#!/usr/bin/python
#-*- coding:utf-8 -*-

from __future__ import division

import cPickle
import numpy as np
import scipy.io as sio


class DataRewriter:
    def __init__(self):
        # 读入数据做初始化
        self.userIndex = cPickle.load(open("PE_userIndex.pkl", 'rb'))
        self.eventIndex = cPickle.load(open("PE_eventIndex.pkl", 'rb'))
        self.userEventScores = sio.mmread("PE_userEventScores").todense()
        self.userSimMatrix = sio.mmread("US_userSimMatrix").todense()
        self.eventPropSim = sio.mmread("EV_eventPropSim").todense()
        self.eventContSim = sio.mmread("EV_eventContSim").todense()
        self.numFriends = sio.mmread("UF_numFriends")
        self.userFriends = sio.mmread("UF_userFriends").todense()
        self.eventPopularity = sio.mmread("EA_eventPopularity").todense()

    def userReco(self, userId, eventId):
        i = self.userIndex[userId]
        j = self.eventIndex[eventId]
        vs = self.userEventScores[:, j]
        sims = self.userSimMatrix[i, :]
        prod = sims * vs
        try:
            return prod[0, 0] - self.userEventScores[i, j]
        except IndexError:
            return 0

    def eventReco(self, userId, eventId):
        i = self.userIndex[userId]
        j = self.eventIndex[eventId]
        js = self.userEventScores[i, :]
        psim = self.eventPropSim[:, j]
        csim = self.eventContSim[:, j]
        pprod = js * psim
        cprod = js * csim
        pscore = 0
        cscore = 0
        try:
            pscore = pprod[0, 0] - self.userEventScores[i, j]
        except IndexError:
            pass
        try:
            cscore = cprod[0, 0] - self.userEventScores[i, j]
        except IndexError:
            pass
        return pscore, cscore

    def userPop(self, userId):
        if self.userIndex.has_key(userId):
            i = self.userIndex[userId]
            try:
                return self.numFriends[0, i]
            except IndexError:
                return 0
        else:
            return 0

    def friendInfluence(self, userId):
        """
        朋友对用户的影响
        主要考虑用户所有的朋友中，有多少是非常喜欢参加各种社交活动/event的
        用户的朋友圈如果都积极参与各种event，可能会对当前用户有一定的影响
        """
        nusers = np.shape(self.userFriends)[1]
        i = self.userIndex[userId]
        return (self.userFriends[i, :].sum(axis=0) / nusers)[0, 0]

    def eventPop(self, eventId):
        """
        本活动本身的热度
        通过参与的人数来界定
        """
        i = self.eventIndex[eventId]
        return self.eventPopularity[i, 0]

    def rewriteData(self, start=1, train=True, header=True):
        """
        把前面user-based协同过滤 和 item-based协同过滤，以及各种热度和影响度作为特征组合在一起
        生成新的训练数据，用于分类器分类使用
        """
        fn = "train.csv" if train else "test.csv"
        fin = open(fn, 'rb')
        fout = open("data_" + fn, 'wb')
        # write output header
        if header:
            ocolnames = ["invited", "user_reco", "evt_p_reco",
                         "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]
            if train:
                ocolnames.append("interested")
                ocolnames.append("not_interested")
            fout.write(",".join(ocolnames) + "\n")
        ln = 0
        for line in fin:
            ln += 1
            if ln < start:
                continue
            cols = line.strip().split(",")
            userId = cols[0]
            eventId = cols[1]
            invited = cols[2]
            if ln % 500 == 0:
                print "%s:%d (userId, eventId)=(%s, %s)" % (fn, ln, userId, eventId)
            user_reco = self.userReco(userId, eventId)
            evt_p_reco, evt_c_reco = self.eventReco(userId, eventId)
            user_pop = self.userPop(userId)
            frnd_infl = self.friendInfluence(userId)
            evt_pop = self.eventPop(eventId)
            ocols = [invited, user_reco, evt_p_reco,
                     evt_c_reco, user_pop, frnd_infl, evt_pop]
            if train:
                ocols.append(cols[4])  # interested
                ocols.append(cols[5])  # not_interested
            fout.write(",".join(map(lambda x: str(x), ocols)) + "\n")
        fin.close()
        fout.close()

    def rewriteTrainingSet(self):
        self.rewriteData(True)

    def rewriteTestSet(self):
        self.rewriteData(False)

dr = DataRewriter()
print "生成训练数据...\n"
dr.rewriteData(train=True, start=2, header=True)
print "生成预测数据...\n"
dr.rewriteData(train=False, start=2, header=True)