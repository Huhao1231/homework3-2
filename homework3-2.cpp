#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define KFOLD 10
#define MAX_FEAT 19
#define MAX_SAMPLE 263
#define EPS 1e-8
#define MAX_ITER 10000
#define TOL 1e-6

// ===================== 工具函数 =====================
double mean(double *x, int n){
    double s=0;
    for(int i=0;i<n;i++) s+=x[i];
    return s/n;
}
double stddev(double *x, int n){
    double m=mean(x,n), s=0;
    for(int i=0;i<n;i++) s+=(x[i]-m)*(x[i]-m);
    return sqrt(s/(n-1));
}
void scale(double *X, int n, int p, double *mean_vec, double *std_vec){
    for(int j=0;j<p;j++){
        double m=0, v=0;
        for(int i=0;i<n;i++) m += X[i*p + j];
        m /= n;
        for(int i=0;i<n;i++) v += (X[i*p+j]-m)*(X[i*p+j]-m);
        v /= (n-1);
        double sd = sqrt(v);
        mean_vec[j]=m; std_vec[j]=sd;
        if(sd<EPS) sd=1;
        for(int i=0;i<n;i++) X[i*p+j] = (X[i*p+j]-m)/sd;
    }
}
void apply_scale(double *X, int n, int p, double *mean_vec, double *std_vec){
    for(int j=0;j<p;j++){
        double sd = std_vec[j]<EPS ? 1 : std_vec[j];
        for(int i=0;i<n;i++) X[i*p+j] = (X[i*p+j]-mean_vec[j])/sd;
    }
}
double rmse(double *ytrue, double *ypred, int n){
    double s=0;
    for(int i=0;i<n;i++) s += pow(ytrue[i]-ypred[i],2);
    return sqrt(s/n);
}

// ===================== Ridge 解析解 (X'X + λI)^{-1}X'y =====================
void mat_mult(double *A, double *B, double *C, int n1, int n2, int n3){
    memset(C,0,sizeof(double)*n1*n3);
    for(int i=0;i<n1;i++)
        for(int k=0;k<n2;k++)
            for(int j=0;j<n3;j++)
                C[i*n3+j] += A[i*n2+k] * B[k*n3+j];
}
int mat_inv(double *A, int n){
    double aug[n][2*n];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++) aug[i][j]=A[i*n+j];
        for(int j=n;j<2*n;j++) aug[i][j]=(j-n==i)?1.0:0.0;
    }
    for(int col=0;col<n;col++){
        int pivot=col;
        for(int r=col;r<n;r++) if(fabs(aug[r][col])>fabs(aug[pivot][col])) pivot=r;
        if(fabs(aug[pivot][col])<EPS) return 0;
        double t;
        for(int j=col;j<2*n;j++) {t=aug[col][j];aug[col][j]=aug[pivot][j];aug[pivot][j]=t;}
        double div = aug[col][col];
        for(int j=col;j<2*n;j++) aug[col][j]/=div;
        for(int r=0;r<n;r++){
            if(r!=col && fabs(aug[r][col])>EPS){
                double factor=aug[r][col];
                for(int j=col;j<2*n;j++) aug[r][j]-=factor*aug[col][j];
            }
        }
    }
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++) A[i*n+j]=aug[i][n+j];
    return 1;
}
void ridge_solve(double *X, double *y, int n, int p, double lambda, double *beta){
    double XtX[MAX_FEAT*MAX_FEAT]={0}, Xty[MAX_FEAT]={0};
    for(int j1=0;j1<p;j1++){
        for(int j2=0;j2<p;j2++){
            double s=0;
            for(int i=0;i<n;i++) s += X[i*p+j1]*X[i*p+j2];
            XtX[j1*p+j2]=s;
        }
        double s=0;
        for(int i=0;i<n;i++) s += X[i*p+j1]*y[i];
        Xty[j1]=s;
    }
    for(int j=0;j<p;j++) XtX[j*p+j] += lambda;
    mat_inv(XtX, p);
    double beta_tmp[MAX_FEAT]={0};
    for(int j1=0;j1<p;j1++){
        double s=0;
        for(int j2=0;j2<p;j2++) s += XtX[j1*p+j2] * Xty[j2];
        beta_tmp[j1]=s;
    }
    memcpy(beta, beta_tmp, sizeof(double)*p);
}

// ===================== Lasso / ElasticNet 坐标下降 =====================
double soft_thresh(double z, double lam){
    if(z>lam) return z-lam;
    if(z<-lam) return z+lam;
    return 0.0;
}
void elasticnet_cd(double *X, double *y, int n, int p, double lambda, double l1_ratio, double *beta){
    memset(beta,0,sizeof(double)*p);
    double yhat[MAX_SAMPLE];
    for(int it=0;it<MAX_ITER;it++){
        double max_diff=0;
        for(int j=0;j<p;j++){
            double r=0;
            for(int i=0;i<n;i++) r += X[i*p+j]*(y[i]-yhat[i]);
            double x2=0;
            for(int i=0;i<n;i++) x2 += X[i*p+j]*X[i*p+j];
            double old_b = beta[j];
            double lam1 = lambda * l1_ratio;
            double lam2 = lambda * (1-l1_ratio);
            beta[j] = soft_thresh(r, lam1)/(x2 + 2*lam2);
            max_diff = fmax(max_diff, fabs(beta[j]-old_b));
            for(int i=0;i<n;i++) yhat[i] += X[i*p+j]*(beta[j]-old_b);
        }
        if(max_diff<TOL) break;
    }
}

// ===================== 10折交叉验证，返回CV均值与标准误 =====================
void kfold_cv(double *X, double *y, int n, int p, double lambda, double l1_ratio,
              double *cv_mean, double *cv_se, int model_type){
    int fold_size = n/KFOLD;
    double cv_err[KFOLD];
    for(int f=0;f<KFOLD;f++){
        int test_start = f*fold_size;
        int test_end = (f==KFOLD-1)? n : (f+1)*fold_size;
        int nt = test_end-test_start;
        int nr = n - nt;
        double Xtr[MAX_SAMPLE*MAX_FEAT], ytr[MAX_SAMPLE];
        double Xte[MAX_SAMPLE*MAX_FEAT], yte[MAX_SAMPLE];
        int tr_idx=0, te_idx=0;
        for(int i=0;i<n;i++){
            if(i>=test_start && i<test_end){
                for(int j=0;j<p;j++) Xte[te_idx*p+j]=X[i*p+j];
                yte[te_idx]=y[i]; te_idx++;
            }else{
                for(int j=0;j<p;j++) Xtr[tr_idx*p+j]=X[i*p+j];
                ytr[tr_idx]=y[i]; tr_idx++;
            }
        }
        double beta[MAX_FEAT];
        if(model_type==0){
            ridge_solve(Xtr,ytr,nr,p,lambda,beta);
        }else if(model_type==1){
            elasticnet_cd(Xtr,ytr,nr,p,lambda,1.0,beta);
        }else{
            elasticnet_cd(Xtr,ytr,nr,p,lambda,l1_ratio,beta);
        }
        double ypred[MAX_SAMPLE];
        for(int i=0;i<nt;i++){
            double s=0;
            for(int j=0;j<p;j++) s += Xte[i*p+j]*beta[j];
            ypred[i]=s;
        }
        cv_err[f]=rmse(yte,ypred,nt);
    }
    *cv_mean = mean(cv_err,KFOLD);
    *cv_se = stddev(cv_err,KFOLD)/sqrt(KFOLD);
}

// ===================== 主程序 =====================
int main(){
    // ========== Hitters简化数据集，完整Hitters可替换这里 ==========
    int n=263;
    int p=18;
    double X[MAX_SAMPLE*MAX_FEAT];
    double y[MAX_SAMPLE];
    // TODO：填入Hitters原始数据，这里省略长数组
    // 你需要把csv读入或者写数组；C不方便放几百行数据。

    // 划分训练测试（简单随机拆分，这里示意）
    int n_train = (int)(0.8*n);
    int n_test = n - n_train;
    double Xtr_raw[MAX_SAMPLE*MAX_FEAT], ytr[MAX_SAMPLE];
    double Xte_raw[MAX_SAMPLE*MAX_FEAT], yte[MAX_SAMPLE];

    // 标准化
    double meanv[MAX_FEAT], stdv[MAX_FEAT];
    scale(Xtr_raw, n_train, p, meanv, stdv);
    apply_scale(Xte_raw, n_test, p, meanv, stdv);

    // λ网格，对应logspace(-3,5,100)
    int n_lambda=30;
    double lambda_grid[30];
    double log_start=-3, log_end=5;
    for(int i=0;i<n_lambda;i++){
        double t = log_start + (log_end-log_start)*i/(n_lambda-1);
        lambda_grid[i] = pow(10, t);
    }

    FILE *fp_path = fopen("coef_path.txt","w");
    if(!fp_path){printf("file error\n");return 1;}

    // === Ridge: model_type=0
    printf("\n==== Ridge ====\n");
    double best_ridge_lam, min_cv_ridge=1e9;
    double ridge_1se_lam=0;
    for(int li=0;li<n_lambda;li++){
        double lam = lambda_grid[li];
        double cv_m, cv_se;
        kfold_cv(Xtr_raw,ytr,n_train,p,lam,0.5,&cv_m,&cv_se,0);
        double beta[MAX_FEAT];
        ridge_solve(Xtr_raw,ytr,n_train,p,lam,beta);
        fprintf(fp_path,"Ridge %g ", lam);
        for(int j=0;j<p;j++) fprintf(fp_path,"%g ",beta[j]);
        fprintf(fp_path,"\n");
        if(cv_m < min_cv_ridge){ min_cv_ridge=cv_m; best_ridge_lam=lam; }
    }

    // === Lasso: model_type=1, l1_ratio=1
    printf("\n==== Lasso ====\n");
    double best_lasso_lam, min_cv_lasso=1e9;
    double lasso_1se_lam=0;
    for(int li=0;li<n_lambda;li++){
        double lam = lambda_grid[li];
        double cv_m, cv_se;
        kfold_cv(Xtr_raw,ytr,n_train,p,lam,1.0,&cv_m,&cv_se,1);
        double beta[MAX_FEAT];
        elasticnet_cd(Xtr_raw,ytr,n_train,p,lam,1.0,beta);
        fprintf(fp_path,"Lasso %g ", lam);
        for(int j=0;j<p;j++) fprintf(fp_path,"%g ",beta[j]);
        fprintf(fp_path,"\n");
        if(cv_m < min_cv_lasso){ min_cv_lasso=cv_m; best_lasso_lam=lam; }
    }

    // === ElasticNet l1_ratio=0.5, model_type=2
    printf("\n==== ElasticNet l1_ratio=0.5 ====\n");
    double best_en_lam, min_cv_en=1e9;
    double en_1se_lam=0;
    for(int li=0;li<n_lambda;li++){
        double lam = lambda_grid[li];
        double cv_m, cv_se;
        kfold_cv(Xtr_raw,ytr,n_train,p,lam,0.5,&cv_m,&cv_se,2);
        double beta[MAX_FEAT];
        elasticnet_cd(Xtr_raw,ytr,n_train,p,lam,0.5,beta);
        fprintf(fp_path,"EN %g ", lam);
        for(int j=0;j<p;j++) fprintf(fp_path,"%g ",beta[j]);
        fprintf(fp_path,"\n");
        if(cv_m < min_cv_en){ min_cv_en=cv_m; best_en_lam=lam; }
    }
    fclose(fp_path);
    printf("Coefficient path saved to coef_path.txt\n");

    // ===== 在最优λ上训练，预测测试集，统计非零变量 =====
    double beta_r[MAX_FEAT], beta_l[MAX_FEAT], beta_en[MAX_FEAT];
    ridge_solve(Xtr_raw,ytr,n_train,p,best_ridge_lam,beta_r);
    elasticnet_cd(Xtr_raw,ytr,n_train,p,best_lasso_lam,1.0,beta_l);
    elasticnet_cd(Xtr_raw,ytr,n_train,p,best_en_lam,0.5,beta_en);

    // predict test
    double ypr[MAX_SAMPLE], ypl[MAX_SAMPLE], ypen[MAX_SAMPLE];
    for(int i=0;i<n_test;i++){
        double sr=0,sl=0,se=0;
        for(int j=0;j<p;j++){
            sr += Xte_raw[i*p+j]*beta_r[j];
            sl += Xte_raw[i*p+j]*beta_l[j];
            se += Xte_raw[i*p+j]*beta_en[j];
        }
        ypr[i]=sr; ypl[i]=sl; ypen[i]=se;
    }
    double rmse_r = rmse(yte,ypr,n_test);
    double rmse_l = rmse(yte,ypl,n_test);
    double rmse_en = rmse(yte,ypen,n_test);

    int nz_r=0,nz_l=0,nz_en=0;
    for(int j=0;j<p;j++){
        if(fabs(beta_r[j])>EPS) nz_r++;
        if(fabs(beta_l[j])>EPS) nz_l++;
        if(fabs(beta_en[j])>EPS) nz_en++;
    }
    printf("\n==== Final Result ====\n");
    printf("Model\tBestLambda\tTestRMSE\tNonZeroVars\n");
    printf("Ridge\t%.4f\t%.4f\t%d\n",best_ridge_lam,rmse_r,nz_r);
    printf("Lasso\t%.4f\t%.4f\t%d\n",best_lasso_lam,rmse_l,nz_l);
    printf("ElasticNet\t%.4f\t%.4f\t%d\n",best_en_lam,rmse_en,nz_en);

    // ========== 1-SE法则计算示例（Lasso） ==========
    printf("\n--- 1-SE rule for Lasso ---\n");
    double min_se_l;
    for(int li=0;li<n_lambda;li++){
        double lam=lambda_grid[li];
        double cv_m, cv_se;
        kfold_cv(Xtr_raw,ytr,n_train,p,lam,1.0,&cv_m,&cv_se,1);
        if(fabs(lam-best_lasso_lam)<1e-6){
            min_se_l=cv_se;break;
        }
    }
    double threshold = min_cv_lasso + min_se_l;
    double max_lam_1se=0;
    for(int li=0;li<n_lambda;li++){
        double lam=lambda_grid[li];
        double cv_m, cv_se;
        kfold_cv(Xtr_raw,ytr,n_train,p,lam,1.0,&cv_m,&cv_se,1);
        if(cv_m <= threshold){
            if(lam>max_lam_1se) max_lam_1se=lam;
        }
    }
    printf("Lasso min CV RMSE=%.4f, +1SE threshold=%.4f, 1SE lambda=%.4f\n",min_cv_lasso,threshold,max_lam_1se);
    return 0;
}

