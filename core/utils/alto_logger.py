import os


class AdaptLogger:
    def __init__(self,args,true_id,id):
        true_id = os.path.splitext(true_id)[0]
        true_id = true_id.lstrip('.datasets/')
        id_ele = true_id.split('/')
        
        self.steps=[str(id),id_ele[0],id_ele[-2],id_ele[-1]]
        self.args = args
        return 
    
    def reserve_step(self,epe):
        self.steps.append(epe)
        return 
    
    def push_line(self):
        txtpath = os.path.join(self.args.output,"valepes.txt")
        with open(txtpath,"a") as f:
            for item in self.steps:
                if isinstance(item,str):
                    f.write(item+" ")
                else:
                    f.write(f"{item:.3f} ")
            f.write("\n")
        return 