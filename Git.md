# 笨办法学Git

### 1.设置

- 名字邮箱

  ```
  $ git config --global user.name "Your Name"
  $ git config --global user.email "your_email@whatever.com"
  ```

### 3.创建项目

- 创建仓库

  ```
  git init
  ```

### 6.暂存更改

- 暂存更改

  ```
  git add .
  ```

### 8.提交更改

- 使用编辑器

  ```
  git commit
  ```

- 使用命令行

  ```
  git commit -m "comment"
  ```

### 10.历史

- 查看历史

  ```
  git log
  ```

- 单行格式

  ```
  git log --pretty=oneline
  ```

- 终极格式

  ```
  git log --pretty=format:'%h %ad | %s%d [%an]' --graph --date=short
  ```

### 11.别名

- 常用别名

  > ./.git/config后添加(项目及级)
  >
  > user/.gitconfig后添加(系统级)

  ```
  [alias]
    co = checkout
    ci = commit
    st = status
    br = branch
    hist = log --pretty=format:'%h %ad | %s%d [%an]' --graph \
    --date=short
    type = cat-file -t
    dump = cat-file -p
  ```

### 12.获得旧版本

- 跳转到旧版本

  ```
  git hist  #查看哈希值
  git checkout <hash>
  ```

- 回到master分支

  ```
  git checkout master
  ```

### 13.给版本打标签

- 标记当前版本

  ```
  git tag v1
  ```

- 标记父提交版本

  ```
  git checkout v1^
  git tag v0
  ```

- 查看标签

  ```
  git tag
  git hist master
  ```

### 14.撤销本地更改(在暂存前)

- 使用 `checkout` 命令来检出 hello.rb 在仓库中的版本。

  ```
  git checkout hello.rb
  ```

### 15.撤销暂存的更改

- 此时已经add过了

  ```
  git reset HEAD hello.rb
  git checkout hello.rb
  ```

### 16.撤销提交的更改

- 创建还原提交

  撤销我们做的最后提交

  ```
  git revert HEAD
  ```

  撤销早期历史中的任意提交

  ```
  git revert <hash>
  ```

  你可以编辑默认的提交信息，或者直接离开。

  本次撤销也会作为一次提交。

### 17.从分支移除提交

- 重置会之前正确的v1版本

  ```
  git tag wrongversion #标记当前版本
  git reset --hard v1
  ```

  `--hard` 参数表示应当更新工作目录以便与新的分支头保持一致。

  错误的提交并没有消失。它们仍然在仓库中。

- 移除错误版本标签

  ```
  git tag -d wrongversion
  git hist --all
  ```

### 19.修正提交

- 修正先前的提交

  ```
  git add .
  git commit --amend -m "new comment"
  ```

### 20.移动文件

- 使用git命令而非系统命令

  ```
  mkdir lib
  git mv hello.rb lib
  ```

  将hello.rb 移动到了新文件夹

### 24.创建分支

- 分两步

  ```
  git branch <branchname>
  git checkout <branchname>
  ```

- 合并成一步

  ```
  git checkout -b greet
  ```

- 查看分支

  ```
  git branch
  git branch -a (查看全部分支)
  ```

### 28.合并

- 合并两个分支

  ```
  git checkout greet
  git merge master
  git hist --all
  ```

- 解决冲突

  手动解决，然后重新add/commit

### 34.变基

不要使用变基(rebase)

### 37.克隆仓库

```
git clone hello cloned_hello
```

### 39.何为Origin

我们看到克隆的仓库知道远程仓库名为 origin。让我们看看是否能获得有关 origin 的更多信息：

```
$ git remote show origin
```

### 42.拉取更改

- 在克隆仓库中

  ```
  git fetch
  ```

  `git fetch` 命令的结果将从远程仓库取得新的提交，但它不会将这些提交合并到本地分支中。

  所以还要合并：

  ```
  git merge origin/master
  ```

- 合并成一步

  ```
  git pull
  ```

### 45.添加跟踪的分支

- 添加跟踪远程分支的本地分支

  ```
  git branch --track greet origin/greet
  ```

- 查看

  ```
  git branch -a
  ```

  

