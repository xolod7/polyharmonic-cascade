import torch
import math
import copy
import time
from sklearn.metrics import roc_auc_score


# Класс, ускоряющий операции с созвездиями гипероктаэдров
class octaedrum:

    def __init__(self, b, c, inp):
        self.n = inp
        k0 = c
        k1 = c - b
        k2 = math.log(2) - 2*b + c
        k4 = 2*(math.log(4) - 2*b) + c
        a1 = k0 - k2
        a2 = k4 - k2
        a3 = k2 - k1*k1/k0
        self.b1 = a1/(a1*a1 - a2*a2)
        self.b2 = -a2/(a1*a1 - a2*a2)
        self.b3 = -a3/((a1 + a2 + 2*inp*a3)*(a1 + a2))
        self.u1 = (k0 + k1*k1*2*inp*(self.b1 + self.b2 + 2*inp*self.b3))/(k0*k0)
        self.u2 = -k1*(self.b1 + self.b2 + 2*inp*self.b3)/k0


    def multi1(self,Y):
        n = self.n
        n2 = 2*n
        y1 = Y[:,0:1,:]
        Y1 = Y[:,1:n+1,:]
        Y2 = Y[:,n+1:,:]
        Y12 = Y[:,1:,:]
        Y21 = torch.cat([Y2, Y1], dim=1)
        ys = torch.sum(Y12,dim=1,keepdim=True)
        h1 = self.u1*y1 + self.u2*ys
        h2 = self.b1*Y12 + self.b2*Y21 + (self.u2*y1 + self.b3*ys).repeat(1,n2,1)
        return torch.cat([h1, h2], dim=1)
    

    def multi2(self,K):
        n = self.n
        n2 = 2*n
        k1 = K[:,:,0:1]
        K1 = K[:,:,1:n+1]
        K2 = K[:,:,n+1:]
        K12 = K[:,:,1:]
        K21 = torch.cat([K2, K1], dim=2)
        ks = torch.sum(K12,dim=2,keepdim=True)
        h1 = self.u1*k1 + self.u2*ks
        h2 = self.b1*K12 + self.b2*K21 + (self.u2*k1 + self.b3*ks).repeat(1,1,n2)
        return torch.cat([h1, h2], dim=2)
    


# Аппроксимирующий слой
class ApproxLayer:

    # Конструктор
    # t, d - параметры автокорреляционной функции
    # inp - количество входов слоя
    # out - количество выходов слоя
    # outs - количество выходов модели
    # invmode - режим без использования обратной матрицы
    def __init__(self, b, c, inp, out, outs, invmode):
        self.b2 = b*2
        self.c = c
        self.inp = inp
        self.invmode = invmode
        self.params = octaedrum(b,c,inp)

        n = 2*inp + 1
        tk = self.stellae(inp)
        obr = torch.tensor([0])
        if not invmode:
            md = self.LinkStellae(tk)
            md += torch.eye(n)
            md = 0.5*md*(torch.log(md) - b*2) + c
            md += b*torch.eye(n)
            obr = torch.linalg.inv(md)
        self.obr = obr

        if out == 0:
            rez = torch.zeros(outs,n,1)
            # rez = (1 - 2*torch.rand(outs, n, 1))/(inp**0.5)
        else:
            if inp==out:
                rez = tk.repeat(outs, 1, 1)
            else:
                rez = (1 - 2*torch.rand(outs, n, out))/(inp**0.5)
        self.rez = rez.double()
        self.Turbine()


    # Решить уравнения
    def Turbine(self):
        if self.invmode:
            self.kof = self.params.multi1(self.rez)
        else:
            self.kof = torch.matmul(self.obr,self.rez)


    # Созвездие точек
    def stellae(self,n):
        return torch.cat([torch.zeros(1,n), -torch.eye(n), torch.eye(n)], dim=0).double()
    

    # Определение матрицы расстояний
    def LinkStellae(self,tx):
        inp = tx.shape[-1]
        m2 = (tx*tx).repeat(1, 1, 1)
        h2 = m2.sum(dim=-1, keepdim = True)
        m2 = (h2 + 1).repeat(1, 1, inp)
        return torch.cat([h2, m2+(tx+tx), m2-(tx+tx)], dim=-1)
    

    def PreparationPars(self, tx):
        psize = 5000
        cdata = tx.shape[-2]
        result = torch.zeros(1,cdata,self.inp*2+1)
        n1 = 0
        for i in range(0,cdata,psize):
            n2 = n1+psize
            if n2>cdata:
                n2 = cdata
            result[:,n1:n2,:] = self.Preparation(tx[n1:n2,:])
            n1 = n2
        return result       
    

    # Специальная подготовка данных первого слоя
    def Preparation(self, tx):
        md = self.LinkStellae(tx)

        f = 0.0000000001
        md[md<=f] = f
        H = 0.5*md*(torch.log(md) - self.b2) + self.c
        if self.invmode:
            return self.params.multi2(H)
        else:
            return torch.matmul(H, self.obr.to(torch.device("cpu")))
        

    # Обработка аппроксимирующим слоем поступающих данных
    def CalcIgnis(self, tx):
        md = self.LinkStellae(tx)
        f = 0.0000000001
        md[md<f] = f
        md = 0.5*md*(torch.log(md) - self.b2) + self.c
        return torch.matmul(md, self.kof)
    

    # Обработка данных первsv слоем
    def Portion1(self, H):
        self.H = H
        return torch.matmul(H, self.rez)
    

    # Обработка данных и вычисление основных матриц для обучения
    def Portion(self, tx):
        self.tx = tx
        md = self.LinkStellae(tx)

        f = 0.0000000001
        md[md<f] = f
        self.X = torch.log(md) - self.b2 + 1

        M = 0.5*md*(self.X - 1) + self.c
        if self.invmode:
            self.H = self.params.multi2(M)
        else:
            self.H = torch.matmul(M, self.obr)
        return torch.matmul(M, self.kof)
    

    # Вычисление матрицы уравнений "нейрокорелята" для слоя
    def Neurocorrelate(self, gr):
        self.gr = gr
        F = torch.matmul(self.H, torch.transpose(self.H, 1, 2))
        G = torch.matmul(gr, torch.transpose(gr, 1, 2))
        return F*G
    

    # Вычисление производных для предыдущего слоя
    def Inaccuracy(self, gr):
        dh = self.X*torch.matmul(gr, torch.transpose(self.kof, 1, 2))
        f1 = self.tx*(dh.sum(dim=-1, keepdim = True).repeat(1,1,self.tx.shape[2]))
        f2 = dh[:,:,1:self.inp+1] - dh[:,:,self.inp+1:]
        return f1 + f2
    

    # Обновление коэффициентов аппроксиматоров
    def Evolution(self, q):
        A = self.gr*(q.repeat(1,1,self.gr.shape[2]))
        self.rez += torch.matmul(torch.transpose(self.H, 1, 2), A)
        self.Turbine()


# Полигармонический каскад
class Collective:
    
    # Конструктор
    # schema - схема каскада (одномерный массив размерности последовательности входов/выходов)
    # type = "double" - тип double, "float" - тип float (при вычислениях)
    # mode = "cpu" или "gpu"
    # func - функция 'r' - регрессия, 'c' - классификация    
    def __init__(self, schema, type, mode, func):
        self.cpu = torch.device("cpu")
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if mode=="gpu" and not torch.cuda.is_available():
            mode = "cpu"
        if mode == "gpu":
            self.device = self.gpu
        else:
            self.device = self.cpu
        self.type = type
        self.mode = mode
        self.func = func
        self.norm = torch.zeros(1,schema[0])
        self.normb = torch.zeros(1,schema[0])
        k = len(schema)
        outs = schema[-1]
        self.outs = outs
        b = 5
        c = 400
        kolich = 0
        porog = 250
        self.lev = []
        for i in range(1,k):
            inp = schema[i-1]
            out = schema[i]
            n = 2*inp + 1
            invmode = n>porog
            if i==k-1:
                kolich += n*outs
                out = 0
            else:
                kolich += out*n*outs
            obj = ApproxLayer(b, c, inp, out, outs, invmode)
            if type == "float":
                obj.kof = obj.kof.float()
                obj.obr = obj.obr.float()
                obj.rez = obj.rez.float()
            obj.kof = obj.kof.to(self.device)
            obj.obr = obj.obr.to(self.device)
            obj.rez = obj.rez.to(self.device)
            self.lev.append(obj)
        self.lev2 = copy.deepcopy(self.lev)
        print("Каскадная система аппроксиматоров")
        print("Количество входов: ", schema[0])
        print("Количество выходов: ", schema[-1])
        print("Слоёв: ", k-1)
        print("Количество настраиваемых параметров: ", kolich)
    

    # Процедура обучения полигармонического каскада
    # (версия, дающая наименьшую ошибку на множестве валидации)
    # возвращается массивы изменения ошибки, характеризующие процесс обучения по эпохам
    # tk/rez - выборка для обучения
    # tk(n х m), 
    # m - размерность входного вектора (количество входов)
    # n - количество векторов для обучения
    # rez(n x p),
    # p - размерность выхода (количество выходов)
    # tx/rz - валидационная выборка 
    # (по которой пошагово будет вычислятся значение ошибки)
    # batch - размер батча
    # epoh - количество эпох
    # mollis - коэффициент "мягкости(сглаживания)" решения систем уравнений
    # подбирается экспериментально, влияет на скорость схождения
    # при больших значениях mollis, процесс обучения может сильно замедлиться
    # при малых значениях mollis, процесс обучения может стать неустойчивым,
    # замедлиться, значения коэффициентов уравнений могут устремиться в
    # бесконечность
    def Disciplina(self, tk, rez, tx, rz, batch, epoh, mollis):
        t1 = time.time()
        if self.func[0] == "c":
            rez = self.DataClass(rez)
            rz = self.DataClass(rz)
        rez = torch.transpose(rez,0,1)
        rez = torch.reshape(rez,(rez.shape[0], rez.shape[1], 1))
        rz = torch.transpose(rz,0,1)
        rz = torch.reshape(rz,(rz.shape[0], rz.shape[1], 1))
        if self.type == "float":
            print("Представление данных: float")
            tk = tk.float()
            rez = rez.float()
            tx = tx.float()
            rz = rz.float()
        else:
            print("Представление данных: double")
            tk = tk.double()
            rez = rez.double()
            tx = tx.double()
            rz = rz.double()
        print("Подготовка")
        if torch.sum(self.norm)==0:
            self.NormirovkaCalc(tk)
        tk = self.Normirovka(tk)
        tx = self.Normirovka(tx)
        tk = self.lev[0].PreparationPars(tk)
        tx = self.lev[0].PreparationPars(tx)
        # tk = tk.to(self.device)
        rez = rez.to(self.device)
        # tx = tx.to(self.device)
        rz = rz.to(self.device)        
        if self.mode == "gpu":
            print("Обучение: gpu")
        else:
            print("Обучение: cpu")
        t2 = time.time()
        print("Время: ", t2 - t1)
        erl = []
        erv = []
        erm = []
        el, ev  = self.OcenkaModeli1(tk, rez, tx, rz)
        if self.func[0] == "r":
            erl.append(el)
            erv.append(ev)
        else:
            if self.outs>1:
                erl.append(100 - el)
                erv.append(100 - ev)
            else:
                erl.append(1-el)
                erv.append(1-ev)
        k = tk.shape[1]
        for j in range(1,epoh+1):
            t1 = time.time()
            ind = torch.randperm(k)
            print("эпоха: ", j)
            for n1 in range(0,k,batch):
                n2 = n1+batch
                if n2>k:
                    break
                id = ind[n1:n2]
                self.Obuchenie(tk[:,id,:].to(self.device), rez[:,id,:], mollis)

            el, ev  = self.OcenkaModeli1(tk, rez, tx, rz)
            t2 = time.time()
            print("Время: ", t2 - t1)
            print("----------")
            if self.func[0] == "r":
                erl.append(el)
                erv.append(ev)
            else:
                if self.outs>1:
                    erl.append(100 - el)
                    erv.append(100 - ev)
                else:
                    erl.append(1-el)
                    erv.append(1-ev)
        
        self.Obnovlenie()
        print("Итоговая модель:")
        self.OcenkaModeli1(tk, rez, tx, rz)
        return erl, erv
    

    # Нужный формат данных для классификации
    def DataClass(self, rez):
        if self.outs>1:
            base = 2*torch.eye(self.outs) - 1
        else:
            base = torch.ones(2,1)
            base[0,0] = -1
        return base[rez,:]
    

    # Вычисление коэффициентов нормировки
    def NormirovkaCalc(self, tk):
        rmx = torch.max(tk, dim=0, keepdim=True).values
        rmn = torch.min(tk, dim=0, keepdim=True).values
        rv = rmx - rmn
        gl = rv==0
        rv[gl] = 1
        self.norm = 1/rv
        self.norm[gl] = 0
        self.normb = -torch.mean(tk*self.norm, dim=0, keepdim=True)

    # def NormirovkaCalc(self, tk):
    #     me = torch.mean(tk*self.norm, dim=0, keepdim=True)
    #     tk2 = tk - me
    #     rv = torch.mean(tk2**2, dim=0, keepdim=True)**0.5
    #     gl = rv==0
    #     rv[gl] = 1
    #     self.norm = 1/rv
    #     self.norm[gl] = 0
    #     self.normb = -torch.mean(tk*self.norm, dim=0, keepdim=True)


    # Нормировка
    def Normirovka(self, tx):
        return tx*self.norm + self.normb


    # Обучение батчу
    # H - входная предобработанная часть обучающей выборки
    # rz - значения на выходе
    # mollis - коэффициент "мягкости(сглаживания)" решения систем уравнений
    def Obuchenie(self, H, rz, mollis):
        k = len(self.lev)
        var = self.lev[0].Portion1(H)
        for i in range(1,k):
            var = self.lev[i].Portion(var)
        gr = rz - var
        gr[gr==0] = 0.00000001
        dv = torch.abs(gr)
        gr = torch.sign(gr)

        Nc = self.lev[-1].Neurocorrelate(gr)
        gr = self.lev[-1].Inaccuracy(gr)
        for i in range(2,k):
            Nc += self.lev[-i].Neurocorrelate(gr)
            gr = self.lev[-i].Inaccuracy(gr)
        Nc += self.lev[0].Neurocorrelate(gr)

        if self.mode == "cpu":
            Nc += mollis*torch.eye(H.shape[1])
        else:
            Nc += mollis*torch.eye(H.shape[1]).to(self.gpu)
        # q = torch.linalg.solve(Nc,dv)
        q = torch.cholesky_solve(dv, torch.linalg.cholesky(Nc))

        for i in range(0,k):
            self.lev[i].Evolution(q)


    # Обработать данные
    def Flamma(self, tx):        
        if self.type == "float":
            tx = tx.float()
        else:
            tx = tx.double()
        tx = self.Normirovka(tx)
        tx = tx.to(self.device)
        var = self.lev[0].CalcIgnis(tx)
        k = len(self.lev)
        for i in range(1,k):
            var = self.lev[i].CalcIgnis(var)
        if self.mode == "gpu":
            var = var.to(self.cpu)
        var = var.reshape(var.shape[0], var.shape[1])
        if self.func[0] == "r":
            return torch.transpose(var,0,1)
        else:
            if var.shape[0] == 1:
                return var.reshape(var.shape[1])
            return torch.argmax(var, dim=0)
    

    # Обработать данные по частям
    def FlammaPars(self,tx,psize):
        k = len(self.lev)
        outs = self.lev[k-1].rez.shape[-1]        
        cdata = tx.shape[-2]
        if self.func[0] == "r":
            result = torch.zeros(cdata,outs)
        else:
            result = torch.zeros(cdata)
        n1 = 0
        for i in range(0,cdata,psize):
            n2 = n1+psize
            if n2>cdata:
                n2 = cdata
            if self.func[0] == "r":
                result[n1:n2,:] = self.Flamma(tx[n1:n2,:])
            else:
                result[n1:n2] = self.Flamma(tx[n1:n2,:])
            n1 = n2
        return result        


    # Обновление сохраняемой модели
    def Obnovlenie(self):
        k = len(self.lev)
        for i in range(0,k):
            self.lev2[i].rez = self.lev[i].rez
            self.lev2[i].kof = self.lev[i].kof
            self.lev2[i].obr = self.lev[i].obr
        self.lev = copy.deepcopy(self.lev2)

        

    # Оценка модели (предварительная/заключительная)
    def OcenkaModeli1(self, tk, rez, tx, rz):
        print()
        vart = self.CalcPurePars(tk,10000)
        var = self.CalcPurePars(tx,10000)
        if self.func[0] == "r":
            evec = torch.mean((rz-var)**2, dim=1)
            erv = torch.mean(evec)**0.5
            erl = torch.mean((rez-vart)**2)**0.5
            erv = erv.to(self.cpu).numpy()
            erl = erl.to(self.cpu).numpy()
            print("ошибка обучения: ")
            print(erl)
            print()
            print("ошибка валидации: ")
            print(erv)
            print()
        else:
            if self.outs>1:
                erv = self.ErrorClass(var, rz).to(self.cpu).numpy()
                erl = self.ErrorClass(vart, rez).to(self.cpu).numpy()            
                print("точность, обучение: ")
                print(100-erl, "%")
                print()
                print("точность, валидация: ")
                print(100-erv, "%")
                print()
            else:
                k = rez.shape[-2]
                auc1 = roc_auc_score(rez.reshape(k).to("cpu"),vart.reshape(k).to("cpu"))
                k = rz.shape[-2]
                auc2 = roc_auc_score(rz.reshape(k).to("cpu"),var.reshape(k).to("cpu"))    
                print("Обучающее множество, ROC AUC: ")
                print(auc1)
                print()
                print("Тестовое множество, ROC AUC: ")
                print(auc2)
                print()
                erl = 1 - auc1
                erv = 1 - auc2
        return erl, erv


    # Вычисление ошибки классификации
    def ErrorClass(self, var, rz):
        if var.shape[0] == 1:
            var = torch.cat([var, -var], dim=0)
            rz = torch.cat([rz, -rz], dim=0)
        num1 = torch.argmax(var, dim=0)
        num2 = torch.argmax(rz, dim=0)
        return 100*torch.sum(num1!=num2)/var.shape[1]
    

    # Обработать выборку
    def CalcPurePars(self, tx, psize):   
        cdata = tx.shape[-2]
        result = torch.zeros(self.outs,cdata,1).to(self.device)
        n1 = 0
        for i in range(0,cdata,psize):
            n2 = n1+psize
            if n2>cdata:
                n2 = cdata
            result[:,n1:n2,:] = self.CalcPure(tx[:,n1:n2,:].to(self.device))
            n1 = n2
        return result           


    # Обработать выборку
    def CalcPure(self, E):
        var = torch.matmul(E, self.lev[0].rez)
        k = len(self.lev)
        for i in range(1,k):
            var = self.lev[i].CalcIgnis(var)
        return var


    # Изменить режим работы
    def Modus(self, type, mode):
        k = len(self.lev)
        if type == "float" and self.type == "double":
            for i in range(0,k):
                self.lev[i].kof = self.lev[i].kof.float()
                self.lev[i].obr = self.lev[i].obr.float()
                self.lev[i].rez = self.lev[i].rez.float()
        if type == "double" and self.type == "float":
            for i in range(0,k):
                self.lev[i].kof = self.lev[i].kof.double()
                self.lev[i].obr = self.lev[i].obr.double()
                self.lev[i].rez = self.lev[i].rez.double()
        self.cpu = torch.device("cpu")
        self.gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if mode=="gpu" and not torch.cuda.is_available():
            mode = "cpu"
        if mode == "gpu" and self.mode == "cpu":
            for i in range(0,k):
                self.lev[i].kof = self.lev[i].kof.to(self.gpu)
                self.lev[i].obr = self.lev[i].obr.to(self.gpu)
                self.lev[i].rez = self.lev[i].rez.to(self.gpu)
        if mode == "cpu" and self.mode == "gpu":
            for i in range(0,k):
                self.lev[i].kof = self.lev[i].kof.to(self.cpu)
                self.lev[i].obr = self.lev[i].obr.to(self.cpu)
                self.lev[i].rez = self.lev[i].rez.to(self.cpu)
        self.type = type
        self.mode = mode
        if mode == "gpu":
            self.device = self.gpu
        else:
            self.device = self.cpu        
        print("Представление данных: ", self.type)
        print("Режим работы: ", self.mode)

        