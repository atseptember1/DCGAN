import os
    
def progressBar(epoch, current, total, dis_loss, gan_loss, barLength=20):
    percent = current * 100 / total
    arrow   = "=" * int(percent/100 * barLength - 1) + ">"
    spaces  = "." * (barLength - len(arrow))
    print(f"Epoch {epoch}: [{arrow}{spaces}] {round(percent,2)}% - Dis Loss: {dis_loss} - Gan Loss: {gan_loss}", end='\r')

def validate_path(path):
    if not(os.path.exists(path)):
        os.makedirs(path)